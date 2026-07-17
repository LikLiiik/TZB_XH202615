#!/usr/bin/env python3
"""
Speaker Diarization + 唤醒人匹配 ASR（适用所有长度音频）

Pipeline:
  1. Diarization (damo/speech_campplus_speaker-diarization_common)
     → 分割 cmd 音频中的不同说话人: [(start, end, speaker_id), ...]
  2. 对每个 speaker，提取 CAMPPlus 声纹，与 kws 唤醒人声纹比对
  3. 只拼接目标说话人的片段
  4. Qwen3-ASR 转录 + CER 评估

注意: 通过 monkey-patch 绕过了原始模型的 5s 最低时长限制，
      使 diarization 适用于所有长度的指令音频。

依赖:
  pip install modelscope funasr hdbscan addict simplejson sortedcontainers datasets
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from funasr import AutoModel
from qwen_asr import Qwen3ASRModel

# Monkey-patch: 绕过 ModelScope diarization 的 5s 最低时长限制
from modelscope.pipelines.audio.segmentation_clustering_pipeline import SegmentationClusteringPipeline

_original_check = SegmentationClusteringPipeline.check_audio_list

def _patched_check(self, audio):
    audio_dur = 0
    for i in range(len(audio)):
        seg = audio[i]
        assert seg[1] >= seg[0], 'Wrong time stamps.'
        audio_dur += seg[1] - seg[0]
    assert audio_dur > 0.3, f'Audio too short: {audio_dur:.1f}s (need > 0.3s)'

SegmentationClusteringPipeline.check_audio_list = _patched_check

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# ---- Config ----
BASE_DIR = Path(__file__).parent
DATASET_DIR = Path("E:/ASR_Models/datasetA")
MODEL_DIR = Path("E:/qwen-asr/Qwen3-ASR-1.7B")
POS_JSONL = DATASET_DIR / "pos.jsonl"

SPK_THRESHOLD = 0.25

PUNCT = re.compile(r'[，。！？、；：""''（）《》【】…—·,\.!\?;:\"\'\(\)\[\]{}<>\s]')


def strip_punct(text):
    return PUNCT.sub("", text)


def compute_cer(ref, hyp):
    ref_c = list(strip_punct(ref))
    hyp_c = list(strip_punct(hyp))
    n, m = len(ref_c), len(hyp_c)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return float(m)
    dp = [[0] * (m + 1) for _ in range(2)]
    dp[0] = list(range(m + 1))
    for i in range(1, n + 1):
        cur, prev = i % 2, 1 - i % 2
        dp[cur][0] = i
        for j in range(1, m + 1):
            cost = 0 if ref_c[i - 1] == hyp_c[j - 1] else 1
            dp[cur][j] = min(dp[prev][j] + 1, dp[cur][j - 1] + 1, dp[prev][j - 1] + cost)
    return dp[n % 2][m] / n


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding(spk_model, audio, sr):
    """提取 CAMPPlus 声纹 (192-dim)"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()
    r = spk_model.generate(input=tmp.name)
    os.unlink(tmp.name)
    return r[0]["spk_embedding"].cpu().numpy().flatten()


def load_models():
    """加载三个模型：Diarization + CAMPPlus + Qwen3-ASR"""
    print("=" * 55)
    print("  加载模型")
    print("=" * 55)

    print("[1/3] 加载 Speaker Diarization 模型 (已 patch 5s 限制)...")
    sd = pipeline(
        Tasks.speaker_diarization,
        model="damo/speech_campplus_speaker-diarization_common",
    )
    print("      Diarization 加载完成 (适用 >= 0.3s 音频)")

    print("[2/3] 加载 CAMPPlus 声纹模型...")
    spk_model = AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        model_revision="v2.0.2",
        disable_update=True,
    )
    print("      CAMPPlus 加载完成")

    print("[3/3] 加载 Qwen3-ASR-1.7B...")
    use_cuda = torch.cuda.is_available()
    asr_model = Qwen3ASRModel.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="cuda:0" if use_cuda else "cpu",
        max_new_tokens=256,
    )
    print(f"      Qwen3-ASR 加载完成 (device={'cuda:0' if use_cuda else 'cpu'})")
    print()
    return sd, spk_model, asr_model


def process_sample(sd, spk_model, asr_model, sample, threshold):
    """
    处理单个样本：Diarization → 说话人匹配 → 提取目标说话人 → ASR
    """
    sid = sample["id"]
    ref = sample["识别文本"]
    kws_path = str(DATASET_DIR / sample["唤醒音频"])
    cmd_path = str(DATASET_DIR / sample["识别音频"])

    kws_audio, sr_k = sf.read(kws_path)
    cmd_audio, sr_c = sf.read(cmd_path)
    duration = len(cmd_audio) / sr_c

    # 1. Diarization
    diar_result = sd(cmd_path)
    segments = diar_result["text"]  # [[start, end, speaker_id], ...]

    if not segments:
        return {
            "id": sid, "ref": ref, "hyp": "[NO_SEGMENTS]", "cer": 1.0,
            "duration": duration, "num_speakers": 0, "target_spk": -1,
        }

    # 2. Group by speaker
    spk_segments = defaultdict(list)
    for start, end, spk_id in segments:
        spk_id = int(spk_id)
        s_start = int(start * sr_c)
        s_end = min(int(end * sr_c), len(cmd_audio))
        spk_segments[spk_id].append((start, end, s_start, s_end))

    # 3. Match each speaker to kws embedding
    kws_emb = get_embedding(spk_model, kws_audio, sr_k)
    best_spk = -1
    best_sim = -1

    for spk_id in spk_segments:
        parts = [cmd_audio[s:e] for _, _, s, e in spk_segments[spk_id]]
        spk_audio = np.concatenate(parts)
        spk_emb = get_embedding(spk_model, spk_audio, sr_c)
        sim = cosine(kws_emb, spk_emb)
        if sim > best_sim:
            best_sim = sim
            best_spk = spk_id

    # 4. Extract target speaker audio + ASR
    if best_spk >= 0 and best_sim >= threshold:
        parts = [cmd_audio[s:e] for _, _, s, e in spk_segments[best_spk]]
        target_audio = np.concatenate(parts)
        r = asr_model.transcribe(audio=(target_audio, sr_c), language="Chinese")
        hyp = r[0].text.strip()
        cer = compute_cer(ref, hyp)
        return {
            "id": sid, "ref": ref, "hyp": hyp, "cer": cer,
            "duration": duration, "num_speakers": len(spk_segments),
            "target_spk": best_spk, "target_sim": best_sim,
            "spk_pass": True,
        }
    else:
        return {
            "id": sid, "ref": ref, "hyp": "[NO_TARGET_SPEAKER]", "cer": 1.0,
            "duration": duration, "num_speakers": len(spk_segments),
            "target_spk": best_spk, "target_sim": best_sim,
            "spk_pass": False,
        }


def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization + 唤醒人匹配 ASR (全长度)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=SPK_THRESHOLD)
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    threshold = args.threshold

    sd, spk_model, asr_model = load_models()

    with open(POS_JSONL, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if limit:
        samples = samples[:limit]

    print("=" * 55)
    print(f"  Diarization + 唤醒人匹配 ASR ({len(samples)} 样本)")
    print(f"  声纹阈值: {threshold}")
    print("=" * 55)

    results = []
    t0 = time.time()

    for idx, s in enumerate(samples):
        try:
            r = process_sample(sd, spk_model, asr_model, s, threshold)
            results.append(r)
        except Exception as e:
            results.append({
                "id": s["id"], "ref": s["识别文本"], "hyp": "[ERROR]",
                "cer": 1.0, "duration": 0, "num_speakers": 0, "target_spk": -1,
            })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(samples):
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - idx - 1) / speed if speed > 0 else 0
            pass_cnt = sum(1 for r in results if r.get("spk_pass"))
            reject_cnt = sum(1 for r in results if not r.get("spk_pass"))
            multi_spk = sum(1 for r in results if r.get("num_speakers", 0) > 1)
            print(f"  {idx + 1}/{len(samples)} | "
                  f"通过:{pass_cnt} 拒绝:{reject_cnt} | "
                  f"多人:{multi_spk} | "
                  f"{speed:.1f}/s | ETA {eta / 60:.1f}min")

    # === Evaluation ===
    print(f"\n{'=' * 55}")
    print(f"  评估报告")
    print(f"{'=' * 55}")
    print(f"  总样本:              {len(results)}")

    pass_results = [r for r in results if r.get("spk_pass")]
    reject_results = [r for r in results if not r.get("spk_pass")]
    multi_spk = [r for r in results if r.get("num_speakers", 0) > 1]

    print(f"  声纹通过:            {len(pass_results)} ({100 * len(pass_results) / len(results):.1f}%)")
    print(f"  声纹拒绝:            {len(reject_results)} ({100 * len(reject_results) / len(results):.1f}%)")
    print(f"  检测到多人对话:      {len(multi_spk)} ({100 * len(multi_spk) / len(results):.1f}%)")

    if pass_results:
        transcribed = [r for r in pass_results if r["hyp"] not in ("[ERROR]", "[NO_SEGMENTS]")]
        cers = [r["cer"] for r in transcribed]
        avg_cer = sum(cers) / len(cers)
        perfect = sum(1 for c in cers if c == 0)
        bad = sum(1 for c in cers if c > 1.0)
        print(f"\n  通过样本 CER (n={len(transcribed)}):")
        print(f"    平均 CER:        {avg_cer * 100:.2f}%")
        print(f"    完美 (0%):       {perfect} ({100 * perfect / len(cers):.1f}%)")
        print(f"    严重错误(>100%): {bad} ({100 * bad / len(cers):.1f}%)")

        # 多人 vs 单人
        single_pass = [r for r in pass_results if r.get("num_speakers", 0) <= 1]
        multi_pass = [r for r in pass_results if r.get("num_speakers", 0) > 1]
        for label, group in [("单人", single_pass), ("多人", multi_pass)]:
            valid = [r for r in group if r["hyp"] not in ("[ERROR]", "[NO_SEGMENTS]")]
            if valid:
                print(f"    {label}: {len(valid)} 条, CER={sum(r['cer'] for r in valid) / len(valid) * 100:.1f}%")

    # 多人样本详情
    if multi_spk:
        print(f"\n  多人样本详情 (前 10):")
        for r in multi_spk[:10]:
            print(f"    [{r['id']}] {r['num_speakers']} speakers, target=spk_{r['target_spk']}, "
                  f"sim={r.get('target_sim', 0):.3f}, CER={r['cer'] * 100:.1f}%")

    output = BASE_DIR / "pos_diarization_result.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  结果保存: {output}")


if __name__ == "__main__":
    main()
