#!/usr/bin/env python3
"""
Speaker Diarization + 唤醒人匹配 ASR

Pipeline:
  1. Diarization (damo/speech_campplus_speaker-diarization_common)
     → 分割 cmd 音频中的不同说话人: [(start, end, speaker_id), ...]
  2. 对每个 speaker，提取 CAMPPlus 声纹，与 kws 唤醒人声纹比对
  3. 只拼接目标说话人的片段
  4. Qwen3-ASR 转录 + CER 评估

对于 <5s 的短音频，diarization 模型不适用，回退到整段声纹验证。

依赖:
  pip install modelscope funasr hdbscan addict simplejson sortedcontainers datasets

模型:
  - damo/speech_campplus_speaker-diarization_common  (说话人日志)
  - iic/speech_campplus_sv_zh-cn_16k-common           (声纹提取)
  - Qwen3-ASR-1.7B                                     (语音识别)
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
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from qwen_asr import Qwen3ASRModel

# ---- Config ----
BASE_DIR = Path(__file__).parent
DATASET_DIR = Path("E:/ASR_Models/datasetA")
MODEL_DIR = Path("E:/qwen-asr/Qwen3-ASR-1.7B")
POS_JSONL = DATASET_DIR / "pos.jsonl"

SPK_THRESHOLD = 0.25
DIARIZATION_MIN_DUR = 5.0  # diarization 模型要求 > 5s

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

    print("[1/3] 加载 Speaker Diarization 模型...")
    sd = pipeline(
        Tasks.speaker_diarization,
        model="damo/speech_campplus_speaker-diarization_common",
    )
    print("      Diarization 加载完成")

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


def process_with_diarization(sd, spk_model, asr_model, cmd_audio, sr, kws_audio, kws_sr, ref, threshold):
    """
    用 Diarization 分割说话人，匹配唤醒人，只转录目标说话人。
    返回 (hyp, cer, diarization_segments, target_speaker_id)
    """
    # 1. Diarization — 需要临时文件
    tmp_cmd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_cmd.name, cmd_audio, sr)
    tmp_cmd.close()
    diar_result = sd(tmp_cmd.name)
    os.unlink(tmp_cmd.name)

    segments = diar_result["text"]  # [[start, end, speaker_id], ...]

    # 2. Group by speaker
    spk_segments = defaultdict(list)
    for start, end, spk_id in segments:
        spk_id = int(spk_id)
        s_start = int(start * sr)
        s_end = min(int(end * sr), len(cmd_audio))
        spk_segments[spk_id].append((start, end, s_start, s_end))

    if not spk_segments:
        return "[NO_SEGMENTS]", 1.0, segments, -1

    # 3. Match each speaker to kws
    kws_emb = get_embedding(spk_model, kws_audio, kws_sr)
    best_spk = -1
    best_sim = -1

    for spk_id in spk_segments:
        parts = [cmd_audio[s:e] for _, _, s, e in spk_segments[spk_id]]
        spk_audio = np.concatenate(parts)
        spk_emb = get_embedding(spk_model, spk_audio, sr)
        sim = cosine(kws_emb, spk_emb)
        if sim > best_sim:
            best_sim = sim
            best_spk = spk_id

    # 4. Extract target speaker's audio
    if best_spk >= 0 and best_sim >= threshold:
        parts = [cmd_audio[s:e] for _, _, s, e in spk_segments[best_spk]]
        target_audio = np.concatenate(parts)

        # 5. ASR
        r = asr_model.transcribe(audio=(target_audio, sr), language="Chinese")
        hyp = r[0].text.strip()
        cer = compute_cer(ref, hyp)
        return hyp, cer, segments, best_spk
    else:
        return "[NO_TARGET_SPEAKER]", 1.0, segments, best_spk


def process_whole_file(spk_model, asr_model, cmd_audio, sr, kws_audio, kws_sr, ref, threshold):
    """
    短音频回退方案：整段声纹验证。
    """
    kws_emb = get_embedding(spk_model, kws_audio, kws_sr)
    cmd_emb = get_embedding(spk_model, cmd_audio, sr)
    sim = cosine(kws_emb, cmd_emb)

    if sim >= threshold:
        r = asr_model.transcribe(audio=(cmd_audio, sr), language="Chinese")
        hyp = r[0].text.strip()
        cer = compute_cer(ref, hyp)
        return hyp, cer, sim, True
    else:
        return "[NON_TARGET_SPEAKER]", 1.0, sim, False


def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization + 唤醒人匹配 ASR")
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
    print("=" * 55)

    results = []
    stats = {"diarization": 0, "whole_file": 0, "too_short": 0,
             "target_found": 0, "target_not_found": 0}
    t0 = time.time()

    for idx, s in enumerate(samples):
        sid = s["id"]
        ref = s["识别文本"]
        kws_path = str(DATASET_DIR / s["唤醒音频"])
        cmd_path = str(DATASET_DIR / s["识别音频"])

        try:
            kws_audio, sr_k = sf.read(kws_path)
            cmd_audio, sr_c = sf.read(cmd_path)
            duration = len(cmd_audio) / sr_c

            if duration >= DIARIZATION_MIN_DUR:
                # Diarization pipeline
                stats["diarization"] += 1
                hyp, cer, segments, target_spk = process_with_diarization(
                    sd, spk_model, asr_model,
                    cmd_audio, sr_c, kws_audio, sr_k, ref, threshold
                )
                results.append({
                    "id": sid, "ref": ref, "hyp": hyp, "cer": cer,
                    "method": "diarization",
                    "duration": duration,
                    "diar_segments": len(segments) if isinstance(segments, list) else 0,
                    "target_spk": target_spk,
                })
                if hyp not in ("[NO_SEGMENTS]", "[NO_TARGET_SPEAKER]"):
                    stats["target_found"] += 1
                else:
                    stats["target_not_found"] += 1
            else:
                # Short audio — whole-file fallback
                stats["whole_file"] += 1
                stats["too_short"] += 1
                hyp, cer, sim, matched = process_whole_file(
                    spk_model, asr_model,
                    cmd_audio, sr_c, kws_audio, sr_k, ref, threshold
                )
                results.append({
                    "id": sid, "ref": ref, "hyp": hyp, "cer": cer,
                    "method": "whole_file",
                    "duration": duration,
                    "spk_sim": sim,
                    "spk_pass": matched,
                })
                if matched:
                    stats["target_found"] += 1
                else:
                    stats["target_not_found"] += 1
        except Exception as e:
            results.append({
                "id": sid, "ref": ref, "hyp": "[ERROR]",
                "cer": 1.0, "method": "error",
                "duration": 0,
            })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(samples):
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - idx - 1) / speed if speed > 0 else 0
            print(f"  {idx + 1}/{len(samples)} | "
                  f"diarization:{stats['diarization']} whole:{stats['whole_file']} | "
                  f"target:{stats['target_found']} other:{stats['target_not_found']} | "
                  f"{speed:.1f}/s | ETA {eta / 60:.1f}min")

    # === Evaluation ===
    print(f"\n{'=' * 55}")
    print(f"  评估报告")
    print(f"{'=' * 55}")
    print(f"  总样本:             {len(results)}")
    print(f"  使用 Diarization:   {stats['diarization']}")
    print(f"  使用整段声纹:       {stats['whole_file']} (< {DIARIZATION_MIN_DUR}s)")
    print(f"  找到目标说话人:     {stats['target_found']}")
    print(f"  未找到/拒识:        {stats['target_not_found']}")

    transcribed = [r for r in results if r["hyp"] not in (
        "[NO_SEGMENTS]", "[NO_TARGET_SPEAKER]", "[NON_TARGET_SPEAKER]", "[ERROR]"
    )]
    if transcribed:
        cers = [r["cer"] for r in transcribed]
        avg_cer = sum(cers) / len(cers)
        perfect = sum(1 for c in cers if c == 0)
        bad = sum(1 for c in cers if c > 1.0)
        print(f"\n  转录样本 CER (n={len(transcribed)}):")
        print(f"    平均 CER:        {avg_cer * 100:.2f}%")
        print(f"    完美 (0%):       {perfect} ({100 * perfect / len(cers):.1f}%)")
        print(f"    严重错误(>100%): {bad} ({100 * bad / len(cers):.1f}%)")

    # By method
    diar_results = [r for r in results if r.get("method") == "diarization"]
    wf_results = [r for r in results if r.get("method") == "whole_file"]
    for label, res in [("Diarization", diar_results), ("Whole-file", wf_results)]:
        if res:
            valid = [r for r in res if r["hyp"] not in (
                "[NO_SEGMENTS]", "[NO_TARGET_SPEAKER]", "[NON_TARGET_SPEAKER]", "[ERROR]"
            )]
            if valid:
                print(f"\n  {label}: {len(valid)} transcribed, "
                      f"CER={sum(r['cer'] for r in valid) / len(valid) * 100:.1f}%")

    # Save
    output = BASE_DIR / "pos_diarization_result.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  结果保存: {output}")


if __name__ == "__main__":
    main()
