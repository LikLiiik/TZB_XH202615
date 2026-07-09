#!/usr/bin/env python3
"""
说话人条件 ASR：用唤醒词(kws)音频提取目标说话人声纹，
只对声纹匹配的命令(cmd)音频进行 ASR 转录。

用法:
    python speaker_aware_asr.py                    # 全量评估
    python speaker_aware_asr.py --limit 20          # 快速测试
    python speaker_aware_asr.py --threshold 0.3     # 自定义阈值
    python speaker_aware_asr.py --pos-only          # 仅评估 pos 集
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from funasr import AutoModel
from qwen_asr import Qwen3ASRModel

# ---- Config ----
BASE_DIR = Path(__file__).parent
DATASET_DIR = Path("E:/ASR_Models/datasetA")
MODEL_DIR = Path("E:/qwen-asr/Qwen3-ASR-1.7B")
POS_JSONL = DATASET_DIR / "pos.jsonl"
NEG_JSONL = DATASET_DIR / "neg.jsonl"
SPK_THRESHOLD = 0.25  # 余弦相似度阈值

PUNCT = re.compile(r'[，。！？、；：""''（）《》【】…—·,\.!\?;:\"\'\(\)\[\]{}<>\s]')


def strip_punct(text: str) -> str:
    return PUNCT.sub("", text)


def compute_cer(ref: str, hyp: str) -> float:
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


def load_models():
    """加载说话人声纹模型 + ASR 模型"""
    print("=" * 55)
    print("  加载模型")
    print("=" * 55)

    print("[1/2] 加载 CAMPPlus 说话人声纹模型...")
    spk_model = AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        model_revision="v2.0.2",
        disable_update=True,
    )
    print("       CAMPPlus 加载完成 (192-dim embeddings)")

    print("[2/2] 加载 Qwen3-ASR-1.7B...")
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    asr_model = Qwen3ASRModel.from_pretrained(
        str(MODEL_DIR),
        dtype=dtype,
        device_map=device,
        max_new_tokens=256,
    )
    print(f"       Qwen3-ASR 加载完成 (device={device})")
    print()
    return spk_model, asr_model


def get_embedding(spk_model, audio_path: str) -> np.ndarray:
    """提取说话人声纹嵌入 (192-dim)"""
    r = spk_model.generate(input=audio_path)
    return r[0]["spk_embedding"].cpu().numpy().flatten()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def process_pos(spk_model, asr_model, samples, threshold, batch_size=32):
    """
    处理 pos 集：声纹验证 + ASR 转录 + CER 评估
    """
    print("=" * 55)
    print(f"  POS 集：说话人条件 ASR ({len(samples)} 样本)")
    print("=" * 55)

    results = []
    t0 = time.time()

    # Phase 1: 批量提取所有 kws 声纹
    print("[Phase 1] 提取唤醒词语音声纹...")
    target_embs = {}
    for s in samples:
        kws_path = str(DATASET_DIR / s["唤醒音频"])
        try:
            target_embs[s["id"]] = get_embedding(spk_model, kws_path)
        except Exception:
            target_embs[s["id"]] = None
    print(f"  完成 {sum(1 for v in target_embs.values() if v is not None)}/{len(samples)} 条")

    # Phase 2: 逐批处理 — 声纹验证 + 批量 ASR
    print("[Phase 2] 声纹验证 + ASR 转录...")
    spk_pass_count = 0
    spk_reject_count = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]

        # Step A: 对本批做声纹验证
        batch_pass = []  # 通过的样本 + 相似度
        for s in batch:
            sid = s["id"]
            temb = target_embs.get(sid)

            if temb is None:
                results.append({
                    "id": sid, "ref": s["识别文本"], "hyp": "[SPK_ERROR]",
                    "cer": 1.0, "spk_pass": False, "spk_sim": 0.0,
                })
                spk_reject_count += 1
                continue

            try:
                cemb = get_embedding(spk_model, str(DATASET_DIR / s["识别音频"]))
            except Exception:
                results.append({
                    "id": sid, "ref": s["识别文本"], "hyp": "[CMD_ERROR]",
                    "cer": 1.0, "spk_pass": False, "spk_sim": 0.0,
                })
                spk_reject_count += 1
                continue

            sim = cosine(temb, cemb)
            if sim < threshold:
                results.append({
                    "id": sid, "ref": s["识别文本"], "hyp": "[NON_TARGET_SPEAKER]",
                    "cer": 1.0, "spk_pass": False, "spk_sim": float(sim),
                })
                spk_reject_count += 1
            else:
                batch_pass.append((s, float(sim)))

        # Step B: 对通过的样本做批量 ASR
        if batch_pass:
            try:
                cmd_paths = [str(DATASET_DIR / s["识别音频"]) for s, _ in batch_pass]
                asr_results = asr_model.transcribe(audio=cmd_paths, language="Chinese")

                for (s, sim), r in zip(batch_pass, asr_results):
                    hyp = r.text.strip()
                    ref = s["识别文本"]
                    cer = compute_cer(ref, hyp)
                    results.append({
                        "id": s["id"], "ref": ref, "hyp": hyp,
                        "cer": cer, "spk_pass": True, "spk_sim": sim,
                    })
                    spk_pass_count += 1
            except Exception as e:
                for s, sim in batch_pass:
                    results.append({
                        "id": s["id"], "ref": s["识别文本"], "hyp": "[ASR_ERROR]",
                        "cer": 1.0, "spk_pass": True, "spk_sim": sim,
                    })

        done = min(i + batch_size, len(samples))
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (len(samples) - done) / speed if speed > 0 else 0
        print(f"  POS: {done}/{len(samples)} ({100*done/len(samples):.0f}%) | "
              f"通过:{spk_pass_count} 拒绝:{spk_reject_count} | {speed:.1f}/s | ETA {eta/60:.1f}min")

    return results


def process_neg(spk_model, asr_model, samples, threshold, batch_size=32):
    """
    处理 neg 集：声纹验证 + 拒识评估
    neg 样本的 "识别文本" 为 null，期望被拒识
    """
    print()
    print("=" * 55)
    print(f"  NEG 集：说话人条件拒识 ({len(samples)} 样本)")
    print("=" * 55)

    results = []
    t0 = time.time()

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]

        for s in batch:
            sid = s["id"]
            kws_path = str(DATASET_DIR / s["唤醒音频"])
            cmd_path = str(DATASET_DIR / s["识别音频"])

            try:
                target_emb = get_embedding(spk_model, kws_path)
                cmd_emb = get_embedding(spk_model, cmd_path)
                sim = cosine(target_emb, cmd_emb)
            except Exception:
                results.append({
                    "id": sid, "hyp": "[ERROR]", "spk_pass": False,
                    "spk_sim": 0.0, "is_rejected": True,
                })
                continue

            if sim < threshold:
                # 非目标说话人 → 正确拒识
                results.append({
                    "id": sid, "hyp": "[NON_TARGET_SPEAKER]",
                    "spk_pass": False, "spk_sim": float(sim),
                    "is_rejected": True,
                })
            else:
                # 声纹通过 → 转录
                try:
                    r = asr_model.transcribe(audio=cmd_path, language="Chinese")
                    hyp = r[0].text.strip()
                    results.append({
                        "id": sid, "hyp": hyp,
                        "spk_pass": True, "spk_sim": float(sim),
                        "is_rejected": (hyp == ""),
                    })
                except Exception:
                    results.append({
                        "id": sid, "hyp": "[ERROR]",
                        "spk_pass": True, "spk_sim": float(sim),
                        "is_rejected": False,
                    })

        done = min(i + batch_size, len(samples))
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (len(samples) - done) / speed if speed > 0 else 0
        spk_pass = sum(1 for r in results if r.get("spk_pass"))
        spk_reject = sum(1 for r in results if not r.get("spk_pass"))
        print(f"  NEG: {done}/{len(samples)} ({100*done/len(samples):.0f}%) | "
              f"声纹通过:{spk_pass} 声纹拒绝:{spk_reject} | {speed:.1f}/s | ETA {eta/60:.1f}min")

    return results


def evaluate(results, tag="POS"):
    """打印评估报告"""
    total = len(results)
    spk_pass = [r for r in results if r.get("spk_pass")]
    spk_reject = [r for r in results if not r.get("spk_pass")]

    print(f"\n{'=' * 55}")
    print(f"  {tag} 评估报告 (阈值={SPK_THRESHOLD})")
    print(f"{'=' * 55}")
    print(f"  总样本:           {total}")
    print(f"  声纹通过:         {len(spk_pass)} ({100*len(spk_pass)/total:.1f}%)")
    print(f"  声纹拒绝:         {len(spk_reject)} ({100*len(spk_reject)/total:.1f}%)")

    if tag == "POS" and spk_pass:
        cers = [r["cer"] for r in spk_pass if r["hyp"] not in ("[ASR_ERROR]", "[CMD_ERROR]")]
        if cers:
            avg_cer = sum(cers) / len(cers)
            perfect = sum(1 for c in cers if c == 0)
            high = sum(1 for c in cers if c > 1.0)
            print(f"  声纹通过样本 CER:")
            print(f"    平均 CER:        {avg_cer*100:.2f}%")
            print(f"    完美 (0%):       {perfect} ({100*perfect/len(cers):.1f}%)")
            print(f"    严重错误(>100%): {high} ({100*high/len(cers):.1f}%)")

        # 与基线对比
        print(f"\n  --- 与基线对比 ---")
        baseline_path = BASE_DIR / "pos_result.jsonl"
        if baseline_path.exists():
            with open(baseline_path, encoding="utf-8") as f:
                baseline = [json.loads(l) for l in f if l.strip()]
            # 只对比声纹通过的样本在原基线中的 CER
            pass_ids = {r["id"] for r in spk_pass}
            baseline_pass = [r for r in baseline if r["id"] in pass_ids]
            if baseline_pass:
                bl_cer = sum(r["cer"] for r in baseline_pass) / len(baseline_pass)
                print(f"  声纹通过的样本在基线中的 CER: {bl_cer*100:.2f}%")
                print(f"  声纹通过样本的 CER:          {avg_cer*100:.2f}%")

    if tag == "NEG":
        total_rejected = sum(1 for r in results if r.get("is_rejected"))
        spk_reject_count = sum(1 for r in spk_reject if r.get("is_rejected"))
        asr_reject_count = sum(1 for r in spk_pass if r.get("is_rejected"))
        print(f"\n  拒识统计:")
        print(f"    声纹拒绝 (非目标说话人): {spk_reject_count}")
        print(f"    ASR 空输出 (正确拒识):    {asr_reject_count}")
        print(f"    总正确拒识:               {total_rejected}")
        print(f"    RR (句准):                {total_rejected/total*100:.2f}%")

    # 声纹相似度分布
    sims = [r["spk_sim"] for r in results if r["spk_sim"] > -99]
    if sims:
        bins = [(-1, 0), (0, 0.15), (0.15, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 1.0)]
        print(f"\n  声纹相似度分布:")
        for lo, hi in bins:
            count = sum(1 for s in sims if lo <= s < hi)
            print(f"    [{lo:.2f}, {hi:.2f}): {count:5d} ({100*count/len(sims):4.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="说话人条件 ASR 评估")
    parser.add_argument("--limit", type=int, default=0, help="限制样本数 (测试用)")
    parser.add_argument("--threshold", type=float, default=SPK_THRESHOLD, help=f"声纹相似度阈值 (默认 {SPK_THRESHOLD})")
    parser.add_argument("--pos-only", action="store_true", help="仅评估 pos 集")
    parser.add_argument("--neg-only", action="store_true", help="仅评估 neg 集")
    args = parser.parse_args()

    threshold = args.threshold
    limit = args.limit if args.limit > 0 else None

    print(f"阈值: {threshold}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 加载模型
    spk_model, asr_model = load_models()

    # ---- POS ----
    if not args.neg_only:
        with open(POS_JSONL, encoding="utf-8") as f:
            pos_samples = [json.loads(l) for l in f if l.strip()]
        if limit:
            pos_samples = pos_samples[:limit]

        pos_results = process_pos(spk_model, asr_model, pos_samples, threshold)
        evaluate(pos_results, "POS")

        # 保存
        output = BASE_DIR / "pos_spk_result.jsonl"
        with open(output, "w", encoding="utf-8") as f:
            for r in pos_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  结果保存至: {output}")

    # ---- NEG ----
    if not args.pos_only:
        with open(NEG_JSONL, encoding="utf-8") as f:
            neg_samples = [json.loads(l) for l in f if l.strip()]
        if limit:
            neg_samples = neg_samples[:limit]

        neg_results = process_neg(spk_model, asr_model, neg_samples, threshold)
        evaluate(neg_results, "NEG")

        output = BASE_DIR / "neg_spk_result.jsonl"
        with open(output, "w", encoding="utf-8") as f:
            for r in neg_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  结果保存至: {output}")

    print(f"\n{'=' * 55}")
    print("  全部完成!")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
