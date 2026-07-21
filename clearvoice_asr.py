#!/usr/bin/env python3
"""
ClearVoice 语音增强 + ASR + 声纹过滤

Pipeline:
  1. ClearVoice FRCRN_SE_16K → 对 cmd 音频降噪增强
  2. CAMPPlus 声纹验证 → 确认增强后音频仍是唤醒人
  3. Qwen3-ASR 转录 → CER 评估

对比：原始音频 ASR vs 增强音频 ASR 的 CER 差异
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from funasr import AutoModel
from qwen_asr import Qwen3ASRModel

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from clearvoice import ClearVoice

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


def load_models():
    print("=" * 55)
    print("  加载模型")
    print("=" * 55)

    print("[1/3] 加载 ClearVoice FRCRN_SE_16K (降噪增强)...")
    cv = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
    print("      FRCRN_SE_16K 加载完成")

    print("[2/3] 加载 CAMPPlus 声纹模型...")
    spk_model = AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        model_revision="v2.0.2", disable_update=True,
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
    print(f"      Qwen3-ASR ({'cuda:0' if use_cuda else 'cpu'})")
    print()
    return cv, spk_model, asr_model


def get_embedding(spk_model, audio, sr):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()
    r = spk_model.generate(input=tmp.name)
    os.unlink(tmp.name)
    return r[0]["spk_embedding"].cpu().numpy().flatten()


def process_sample(cv, spk_model, asr_model, sample, threshold):
    """ClearVoice 增强 → 声纹验证 → ASR"""
    sid = sample["id"]
    ref = sample["识别文本"]
    kws_path = str(DATASET_DIR / sample["唤醒音频"])
    cmd_path = str(DATASET_DIR / sample["识别音频"])

    try:
        kws_audio, sr_k = sf.read(kws_path)
        cmd_audio, sr_c = sf.read(cmd_path)
    except Exception:
        return None

    duration = len(cmd_audio) / sr_c

    # 1. 原始音频 ASR（基线）
    try:
        r_orig = asr_model.transcribe(audio=cmd_path, language="Chinese")
        hyp_orig = r_orig[0].text.strip()
        cer_orig = compute_cer(ref, hyp_orig)
    except Exception:
        hyp_orig, cer_orig = "[ERROR]", 1.0

    # 2. ClearVoice 增强
    try:
        enhanced = cv(input_path=cmd_path, online_write=False)
        enhanced_audio = np.squeeze(enhanced)
    except Exception:
        return {
            "id": sid, "ref": ref, "cer_orig": cer_orig, "cer_enhanced": cer_orig,
            "hyp_orig": hyp_orig, "hyp_enhanced": "[ENHANCE_ERROR]",
            "duration": duration, "method": "enhance_error",
        }

    # 3. 声纹验证（增强后仍是唤醒人？）
    try:
        cmd_emb = get_embedding(spk_model, cmd_audio, sr_c)
        enh_emb = get_embedding(spk_model, enhanced_audio, sr_c)
        kws_emb = get_embedding(spk_model, kws_audio, sr_k)
        sim_cmd = cosine(kws_emb, cmd_emb)
        sim_enh = cosine(kws_emb, enh_emb)
    except Exception:
        sim_cmd, sim_enh = 0, 0

    # 4. 增强音频 ASR
    spk_pass = sim_enh >= threshold
    if spk_pass:
        try:
            r_enh = asr_model.transcribe(audio=(enhanced_audio, sr_c), language="Chinese")
            hyp_enhanced = r_enh[0].text.strip()
            cer_enhanced = compute_cer(ref, hyp_enhanced)
        except Exception:
            hyp_enhanced, cer_enhanced = "[ERROR]", 1.0
        method = "enhanced"
    else:
        hyp_enhanced, cer_enhanced = "[SPK_REJECT]", 1.0
        method = "spk_reject"

    return {
        "id": sid, "ref": ref,
        "cer_orig": cer_orig, "hyp_orig": hyp_orig,
        "cer_enhanced": cer_enhanced, "hyp_enhanced": hyp_enhanced,
        "duration": duration, "method": method,
        "sim_orig": float(sim_cmd), "sim_enhanced": float(sim_enh),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=SPK_THRESHOLD)
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    threshold = args.threshold

    cv, spk_model, asr_model = load_models()

    with open(POS_JSONL, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if limit:
        samples = samples[:limit]

    print("=" * 55)
    print(f"  ClearVoice 增强 + 声纹 + ASR")
    print(f"  {len(samples)} 样本 | 阈值={threshold}")
    print("=" * 55)

    results = []
    t0 = time.time()

    for idx, s in enumerate(samples):
        r = process_sample(cv, spk_model, asr_model, s, threshold)
        if r:
            results.append(r)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - idx - 1) / speed if speed > 0 else 0
            enhanced = sum(1 for r in results if r.get("method") == "enhanced")
            rejected = sum(1 for r in results if r.get("method") == "spk_reject")
            print(f"  {idx + 1}/{len(samples)} | "
                  f"增强:{enhanced} 拒绝:{rejected} | "
                  f"{speed:.1f}/s | ETA {eta / 60:.1f}min")

    # === 评估 ===
    print(f"\n{'=' * 55}")
    print(f"  评估报告")
    print(f"{'=' * 55}")

    enhanced_results = [r for r in results if r.get("method") == "enhanced"]
    rejected_results = [r for r in results if r.get("method") == "spk_reject"]

    print(f"  总样本:              {len(results)}")
    print(f"  增强+声纹通过:       {len(enhanced_results)} ({100*len(enhanced_results)/len(results):.1f}%)")
    print(f"  增强+声纹拒绝:       {len(rejected_results)} ({100*len(rejected_results)/len(results):.1f}%)")

    # 原始 vs 增强 CER
    valid = [r for r in enhanced_results if r["cer_orig"] < 1.0 and r["cer_enhanced"] < 1.0]
    if valid:
        orig_cers = [r["cer_orig"] for r in valid]
        enh_cers = [r["cer_enhanced"] for r in valid]
        print(f"\n  有效增强样本 ({len(valid)}):")
        print(f"    原始音频 CER:     {sum(orig_cers)/len(orig_cers)*100:.2f}%")
        print(f"    增强音频 CER:     {sum(enh_cers)/len(enh_cers)*100:.2f}%")

        improved = sum(1 for r in valid if r["cer_enhanced"] < r["cer_orig"])
        worsened = sum(1 for r in valid if r["cer_enhanced"] > r["cer_orig"])
        unchanged = sum(1 for r in valid if r["cer_enhanced"] == r["cer_orig"])
        print(f"    改善: {improved}, 变差: {worsened}, 不变: {unchanged}")

    # Low / High ID
    for label, id_range in [("Low  (<2000)", (0, 2000)), ("High (>=2000)", (2000, 9999))]:
        group = [r for r in enhanced_results if id_range[0] <= r["id"] < id_range[1]]
        if group:
            orig = [r["cer_orig"] for r in group]
            enh = [r["cer_enhanced"] for r in group]
            print(f"\n  {label}: {len(group)} 条")
            print(f"    原始: {sum(orig)/len(orig)*100:.1f}% → 增强: {sum(enh)/len(enh)*100:.1f}%")

    output = BASE_DIR / "pos_clearvoice_enhance_result.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  结果保存: {output}")


if __name__ == "__main__":
    main()
