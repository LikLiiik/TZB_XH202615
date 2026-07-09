#!/usr/bin/env python3
"""
分段声纹匹配 + ASR：对 cmd 音频按语音段分割，逐段比对唤醒人声纹，
只保留匹配的语音段进行 ASR 转录。

管线的核心区别：
- speaker_aware_asr.py: 整段 cmd vs kws 声纹比较 → 二元通过/拒绝
- segment_speaker_asr.py:  cmd 分段 → 每段 vs kws 声纹 → 拼接匹配段 → ASR
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from funasr import AutoModel
from qwen_asr import Qwen3ASRModel

BASE_DIR = Path(__file__).parent
DATASET_DIR = Path("E:/ASR_Models/datasetA")
MODEL_DIR = Path("E:/qwen-asr/Qwen3-ASR-1.7B")
POS_JSONL = DATASET_DIR / "pos.jsonl"

SPK_THRESHOLD = 0.25  # 声纹匹配阈值

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


def energy_vad(audio, sr, frame_ms=25, energy_threshold=0.02, min_silence_ms=200, min_speech_ms=100):
    """
    基于短时能量的简单 VAD。
    返回语音段列表 [(start_sample, end_sample), ...]
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len // 2  # 50% overlap
    min_silence_frames = int(min_silence_ms / (frame_ms / 2))
    min_speech_frames = int(min_speech_ms / (frame_ms / 2))

    # 计算每帧能量
    energy = []
    for i in range(0, len(audio) - frame_len, hop_len):
        frame = audio[i : i + frame_len]
        energy.append(np.sqrt(np.mean(frame ** 2)))

    energy = np.array(energy)
    threshold = energy_threshold * np.max(energy) if np.max(energy) > 0 else energy_threshold

    # 找到高于阈值的帧
    is_speech = energy > threshold

    # 合并相邻语音段
    segments = []
    in_speech = False
    start_frame = 0
    speech_count = 0
    silence_count = 0

    for i, s in enumerate(is_speech):
        if s:
            if not in_speech:
                start_frame = i
                in_speech = True
            speech_count += 1
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    if speech_count >= min_speech_frames:
                        start_sample = start_frame * hop_len
                        end_sample = (i - silence_count + 1) * hop_len + frame_len
                        segments.append((max(0, start_sample), min(len(audio), end_sample)))
                    in_speech = False
                    speech_count = 0

    if in_speech and speech_count >= min_speech_frames:
        start_sample = start_frame * hop_len
        end_sample = len(audio)
        segments.append((max(0, start_sample), min(len(audio), end_sample)))

    return segments


def get_embedding(spk_model, audio, sr=16000):
    """从 numpy audio 提取说话人声纹"""
    r = spk_model.generate(input=(audio, sr))
    return r[0]["spk_embedding"].cpu().numpy().flatten()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def filter_target_segments(spk_model, cmd_audio, sr, kws_audio, kws_sr, threshold):
    """
    对 cmd 音频分段，保留与 kws 声纹匹配的段。
    返回匹配段的拼接音频，如果无匹配段则返回 None。
    """
    # 1. VAD 分段
    segments = energy_vad(cmd_audio, sr)
    if not segments:
        return None, 0, 0

    # 2. 提取 kws 声纹
    try:
        kws_emb = get_embedding(spk_model, kws_audio, kws_sr)
    except Exception:
        return None, 0, len(segments)

    # 3. 逐段提取声纹并比较
    matched_segments = []
    matched_count = 0
    total_segments = len(segments)

    for start, end in segments:
        seg_audio = cmd_audio[start:end]
        if len(seg_audio) < sr * 0.1:  # 跳过 <0.1s 的段
            continue
        try:
            seg_emb = get_embedding(spk_model, seg_audio, sr)
            sim = cosine(kws_emb, seg_emb)
            if sim >= threshold:
                matched_segments.append(seg_audio)
                matched_count += 1
        except Exception:
            continue

    if not matched_segments:
        return None, 0, total_segments

    # 4. 拼接匹配段
    combined = np.concatenate(matched_segments) if len(matched_segments) > 1 else matched_segments[0]
    return combined, matched_count, total_segments


def load_models():
    print("=" * 55)
    print("  加载模型")
    print("=" * 55)
    print("[1/2] 加载 CAMPPlus 声纹模型...")
    spk_model = AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        model_revision="v2.0.2",
        disable_update=True,
    )
    print("      CAMPPlus 加载完成")

    print("[2/2] 加载 Qwen3-ASR-1.7B...")
    use_cuda = torch.cuda.is_available()
    asr_model = Qwen3ASRModel.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="cuda:0" if use_cuda else "cpu",
        max_new_tokens=256,
    )
    print(f"      Qwen3-ASR 加载完成")
    print()
    return spk_model, asr_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=SPK_THRESHOLD)
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    threshold = args.threshold

    spk_model, asr_model = load_models()

    with open(POS_JSONL, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if limit:
        samples = samples[:limit]

    # 加载基线结果用于对比
    baseline_map = {}
    baseline_path = BASE_DIR / "pos_result.jsonl"
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            baseline_map = {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}

    # 加载 spk_filter 结果用于对比
    spk_filter_map = {}
    spk_filter_path = BASE_DIR / "pos_spk_result.jsonl"
    if spk_filter_path.exists():
        with open(spk_filter_path, encoding="utf-8") as f:
            spk_filter_map = {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}

    print("=" * 55)
    print(f"  分段声纹匹配 ASR ({len(samples)} 样本, 阈值={threshold})")
    print("=" * 55)

    results = []
    full_pass = 0  # 整段通过(无需分段)
    partial_pass = 0  # 部分段通过
    all_reject = 0  # 全部拒绝
    t0 = time.time()

    for idx, s in enumerate(samples):
        sid = s["id"]
        kws_path = str(DATASET_DIR / s["唤醒音频"])
        cmd_path = str(DATASET_DIR / s["识别音频"])
        ref = s["识别文本"]

        try:
            kws_audio, kws_sr = sf.read(kws_path)
            cmd_audio, cmd_sr = sf.read(cmd_path)

            # 确保采样率一致
            if cmd_sr != 16000:
                cmd_audio = sf.resample(cmd_audio, cmd_sr, 16000)  # simplified
                cmd_sr = 16000
            if kws_sr != 16000:
                kws_sr = 16000

            # 分段声纹匹配
            filtered_audio, matched, total = filter_target_segments(
                spk_model, cmd_audio, cmd_sr, kws_audio, kws_sr, threshold
            )

            if filtered_audio is None:
                # 无匹配段
                all_reject += 1
                results.append({
                    "id": sid, "ref": ref, "hyp": "[NO_TARGET_SEGMENT]",
                    "cer": 1.0, "seg_total": total, "seg_matched": 0,
                })
            elif matched == total and total == 1:
                # 只有一段且匹配 — 等同于整段通过
                full_pass += 1
                r = asr_model.transcribe(audio=(filtered_audio, cmd_sr), language="Chinese")
                hyp = r[0].text.strip()
                results.append({
                    "id": sid, "ref": ref, "hyp": hyp,
                    "cer": compute_cer(ref, hyp),
                    "seg_total": total, "seg_matched": matched,
                })
            else:
                # 多段中匹配了部分 — 只转录匹配段
                partial_pass += 1
                r = asr_model.transcribe(audio=(filtered_audio, cmd_sr), language="Chinese")
                hyp = r[0].text.strip()
                results.append({
                    "id": sid, "ref": ref, "hyp": hyp,
                    "cer": compute_cer(ref, hyp),
                    "seg_total": total, "seg_matched": matched,
                })
        except Exception as e:
            results.append({
                "id": sid, "ref": ref, "hyp": "[ERROR]",
                "cer": 1.0, "seg_total": 0, "seg_matched": 0,
            })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - idx - 1) / speed if speed > 0 else 0
            print(f"  {idx+1}/{len(samples)} | 全通过:{full_pass} 部分:{partial_pass} 全拒:{all_reject} | "
                  f"{speed:.1f}/s | ETA {eta/60:.1f}min")

    # === 评估 ===
    print(f"\n{'=' * 55}")
    print(f"  评估报告")
    print(f"{'=' * 55}")
    print(f"  总样本:            {len(results)}")
    print(f"  全段通过:          {full_pass} ({100*full_pass/len(results):.1f}%)")
    print(f"  部分段通过:        {partial_pass} ({100*partial_pass/len(results):.1f}%)")
    print(f"  全部拒绝(无匹配段): {all_reject} ({100*all_reject/len(results):.1f}%)")

    # CER 对比
    transcribed = [r for r in results if r["hyp"] not in ("[NO_TARGET_SEGMENT]", "[ERROR]")]
    if transcribed:
        cers = [r["cer"] for r in transcribed]
        avg_cer = sum(cers) / len(cers)
        perfect = sum(1 for c in cers if c == 0)
        bad = sum(1 for c in cers if c > 1.0)
        print(f"\n  转录样本 CER (n={len(transcribed)}):")
        print(f"    平均 CER:        {avg_cer*100:.2f}%")
        print(f"    完美 (0%):       {perfect} ({100*perfect/len(cers):.1f}%)")
        print(f"    严重错误(>100%): {bad} ({100*bad/len(cers):.1f}%)")

    # 与基线和 spk_filter 对比
    print(f"\n  --- 三方对比 ---")
    all_cer_base = sum(r["cer"] for r in baseline_map.values()) / len(baseline_map) if baseline_map else 0
    all_cer_spk = sum(r["cer"] for r in spk_filter_map.values()) / len(spk_filter_map) if spk_filter_map else 0
    all_cer_seg = sum(r["cer"] for r in results) / len(results)
    print(f"  基线 (无过滤):          {all_cer_base*100:.1f}%")
    print(f"  整段声纹过滤 (spk_filter): {all_cer_spk*100:.1f}%")
    print(f"  分段声纹匹配 (segment):    {all_cer_seg*100:.1f}%")

    # 保存
    output = BASE_DIR / "pos_seg_result.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  结果保存至: {output}")


if __name__ == "__main__":
    main()
