#!/usr/bin/env python3
"""
Qwen3-ASR-1.7B 对 datasetA 进行语音识别推理
- pos.jsonl: 正样本，用于评估 CER (Character Error Rate)
- neg.jsonl: 负样本（拒识测试），"识别文本" 为 null

用法:
    python asr_inference.py                    # 运行全部推理 + 评估
    python asr_inference.py --pos-only         # 仅推理 pos 集
    python asr_inference.py --neg-only         # 仅推理 neg 集
    python asr_inference.py --eval-only        # 仅评估 (已有结果文件时)
    python asr_inference.py --limit 10         # 仅推理前10条 (测试用)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

# ============================================================
# 配置
# ============================================================
BASE_DIR = Path(__file__).parent
DATASET_DIR = Path("E:/ASR_Models/datasetA")  # 音频文件实际路径
POS_JSONL = DATASET_DIR / "pos.jsonl"
NEG_JSONL = DATASET_DIR / "neg.jsonl"
POS_RESULT = BASE_DIR / "pos_result.jsonl"
NEG_RESULT = BASE_DIR / "neg_result.jsonl"

# 模型配置
MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
# 如果本地已下载，可以设置本地路径
# MODEL_NAME = "./Qwen3-ASR-1.7B"


def load_jsonl(file_path, limit=None):
    """读取 JSONL 文件"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


import re

# 中英文标点符号（参考文本不带标点，但模型可能输出标点）
PUNCT_PATTERN = re.compile(
    r'[，。！？、；：""''（）《》【】…—·,\.!\?;:\"\'\(\)\[\]{}<>\s]'
)


def strip_punctuation(text: str) -> str:
    """去除中英文标点符号和空白"""
    return PUNCT_PATTERN.sub("", text)


def compute_cer(reference: str, hypothesis: str, strip_punct: bool = True) -> float:
    """
    计算字符错误率 (Character Error Rate)
    CER = (S + D + I) / N
    使用编辑距离 (Levenshtein distance)

    Args:
        reference: 参考文本
        hypothesis: 识别文本
        strip_punct: 是否先去除标点符号再比较（默认 True，因为参考文本不带标点）
    """
    if strip_punct:
        ref_chars = list(strip_punctuation(reference))
        hyp_chars = list(strip_punctuation(hypothesis))
    else:
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))

    n = len(ref_chars)
    m = len(hyp_chars)

    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return float(m)

    # DP 计算编辑距离（空间优化）
    dp = [[0] * (m + 1) for _ in range(2)]
    dp[0] = list(range(m + 1))

    for i in range(1, n + 1):
        cur = i % 2
        prev = 1 - cur
        dp[cur][0] = i
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp[cur][j] = min(
                dp[prev][j] + 1,       # 删除
                dp[cur][j - 1] + 1,    # 插入
                dp[prev][j - 1] + cost, # 替换
            )

    distance = dp[n % 2][m]
    return distance / n


def load_model(model_name: str, device: str = "cuda:0"):
    """加载 Qwen3-ASR 模型"""
    print(f"[模型] 正在加载模型: {model_name}")

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    if use_cuda:
        print(f"[模型] 使用 CUDA GPU 推理")
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
            max_new_tokens=256,
        )
    else:
        if device.startswith("cuda"):
            print("[模型] CUDA 不可用，降级为 CPU 推理 (速度较慢)")
        else:
            print("[模型] 使用 CPU 推理")
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="cpu",
            max_new_tokens=256,
        )

    print(f"[模型] 加载完成")
    return model


def run_inference_on_pos(model, samples, dataset_dir: Path):
    """对 pos 样本进行 ASR 推理"""
    print(f"\n[推理] 开始处理 pos 集，共 {len(samples)} 条")

    results = []
    audio_paths = []
    labels = []

    # 检查音频文件是否存在
    missing = []
    for s in samples:
        audio_path = dataset_dir / s["识别音频"]
        if not audio_path.exists():
            missing.append(str(audio_path))

    if missing:
        print(f"[警告] 缺少 {len(missing)} 个音频文件!")
        for p in missing[:5]:
            print(f"  - {p}")
        if len(missing) > 5:
            print(f"  ... 等 {len(missing) - 5} 个")
        print(f"[警告] 将跳过缺失文件，仅处理存在的文件")

    # 收集有效的样本
    valid_samples = []
    valid_audio_paths = []
    for s in samples:
        audio_path = dataset_dir / s["识别音频"]
        if audio_path.exists():
            valid_samples.append(s)
            valid_audio_paths.append(str(audio_path))

    if not valid_samples:
        print("[错误] 没有找到任何可用音频文件!")
        return []

    print(f"[推理] 可用音频: {len(valid_samples)}/{len(samples)} 条")

    batch_size = 16  # 批量推理
    total = len(valid_audio_paths)

    # 使用 transcribe 批量处理
    model_start = time.time()
    for i in range(0, total, batch_size):
        batch_paths = valid_audio_paths[i : i + batch_size]
        batch_samples = valid_samples[i : i + batch_size]

        try:
            batch_results = model.transcribe(
                audio=batch_paths,
                language="Chinese",
            )

            for j, r in enumerate(batch_results):
                hyp_text = r.text.strip() if r.text else ""
                ref_text = batch_samples[j]["识别文本"]
                sample_cer = compute_cer(ref_text, hyp_text)
                results.append({
                    "id": batch_samples[j]["id"],
                    "ref": ref_text,
                    "hyp": hyp_text,
                    "cer": sample_cer,
                })

        except Exception as e:
            print(f"[错误] batch {i // batch_size}: {e}")
            # 逐条重试
            for j, (path, sample) in enumerate(zip(batch_paths, batch_samples)):
                try:
                    r = model.transcribe(audio=path, language="Chinese")
                    hyp_text = r[0].text.strip() if r[0].text else ""
                    ref_text = sample["识别文本"]
                    sample_cer = compute_cer(ref_text, hyp_text)
                    results.append({
                        "id": sample["id"],
                        "ref": ref_text,
                        "hyp": hyp_text,
                        "cer": sample_cer,
                    })
                except Exception as e2:
                    print(f"[错误] 样本 {sample['id']}: {e2}")
                    results.append({
                        "id": sample["id"],
                        "ref": sample["识别文本"],
                        "hyp": "[ERROR]",
                        "cer": 1.0,
                    })

        if (i + batch_size) % 100 == 0 or (i + batch_size) >= total:
            elapsed = time.time() - model_start
            done = min(i + batch_size, total)
            speed = done / elapsed if elapsed > 0 else 0
            print(f"  进度: {done}/{total} ({100*done/total:.1f}%), 速度: {speed:.1f} 条/秒")

    elapsed = time.time() - model_start
    print(f"[推理] pos 集完成，耗时: {elapsed:.1f}秒")

    return results


def run_inference_on_neg(model, samples, dataset_dir: Path):
    """对 neg 样本进行 ASR 推理"""
    print(f"\n[推理] 开始处理 neg 集，共 {len(samples)} 条")

    results = []
    valid_paths = []
    valid_samples = []

    for s in samples:
        audio_path = dataset_dir / s["识别音频"]
        if audio_path.exists():
            valid_paths.append(str(audio_path))
            valid_samples.append(s)

    if not valid_samples:
        print("[错误] 没有找到任何可用 neg 音频文件!")
        return []

    print(f"[推理] 可用音频: {len(valid_samples)}/{len(samples)} 条")

    batch_size = 16
    total = len(valid_paths)
    model_start = time.time()

    for i in range(0, total, batch_size):
        batch_paths = valid_paths[i : i + batch_size]
        batch_samples = valid_samples[i : i + batch_size]

        try:
            batch_results = model.transcribe(
                audio=batch_paths,
                language="Chinese",
            )
            for j, r in enumerate(batch_results):
                hyp_text = r.text.strip() if r.text else ""
                results.append({
                    "id": batch_samples[j]["id"],
                    "ref": None,  # neg 样本无参考文本
                    "hyp": hyp_text,
                    "is_rejected": (hyp_text == ""),  # 空文本 = 正确拒识
                })
        except Exception as e:
            print(f"[错误] neg batch {i // batch_size}: {e}")
            for sample in batch_samples:
                results.append({
                    "id": sample["id"],
                    "ref": None,
                    "hyp": "[ERROR]",
                    "is_rejected": False,
                })

        if (i + batch_size) % 100 == 0 or (i + batch_size) >= total:
            done = min(i + batch_size, total)
            print(f"  进度: {done}/{total} ({100*done/total:.1f}%)")

    elapsed = time.time() - model_start
    print(f"[推理] neg 集完成，耗时: {elapsed:.1f}秒")
    return results


def evaluate_pos(results):
    """评估 pos 集: 计算平均 CER"""
    if not results:
        print("[评估] pos 集无结果，跳过")
        return

    valid_results = [r for r in results if r["cer"] < 1.0 or r["hyp"] != "[ERROR]"]
    cers = [r["cer"] for r in results]
    avg_cer = sum(cers) / len(cers) if cers else 0

    print(f"\n{'='*60}")
    print(f"  POS 集 CER 评估结果")
    print(f"{'='*60}")
    print(f"  总样本数:           {len(results)}")
    print(f"  有效样本数:         {len(valid_results)}")
    print(f"  平均 CER:           {avg_cer*100:.2f}%")
    print(f"  最低 CER (最好):    {min(cers)*100:.2f}%")
    print(f"  最高 CER (最差):    {max(cers)*100:.2f}%")

    # CER 分布
    bins = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, float("inf")]
    bin_labels = ["0%-5%", "5%-10%", "10%-20%", "20%-50%", "50%-100%", "100%+"]
    print(f"\n  CER 分布:")
    for low, high, label in zip(bins[:-1], bins[1:], bin_labels):
        count = sum(1 for c in cers if low <= c < high)
        print(f"    {label}:  {count} ({100*count/len(cers):.1f}%)")

    # 打印几个样例
    print(f"\n  样例展示 (前5条):")
    for r in results[:5]:
        print(f"    [{r['id']}] REF: {r['ref']}")
        print(f"    [{r['id']}] HYP: {r['hyp']}")
        print(f"    [{r['id']}] CER: {r['cer']*100:.2f}%")
        print()

    # 打印最差的几条
    print(f"\n  最差样本 (CER最高的5条):")
    worst = sorted(results, key=lambda x: x["cer"], reverse=True)[:5]
    for r in worst:
        print(f"    [{r['id']}] REF: {r['ref']}")
        print(f"    [{r['id']}] HYP: {r['hyp']}")
        print(f"    [{r['id']}] CER: {r['cer']*100:.2f}%")
        print()

    return avg_cer


def evaluate_neg(results):
    """评估 neg 集: 计算 RR (Rejection Rate / 句准)"""
    if not results:
        print("[评估] neg 集无结果，跳过")
        return

    total = len(results)
    rejected = sum(1 for r in results if r["is_rejected"])
    rr = rejected / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  NEG 集 RR (Rejection Rate) 评估结果")
    print(f"{'='*60}")
    print(f"  总样本数:           {total}")
    print(f"  正确拒识数:         {rejected}")
    print(f"  RR (句准):          {rr*100:.2f}%")

    # 统计拒识失败样本 (输出非空文本)
    failed = [r for r in results if not r["is_rejected"]]
    print(f"  拒识失败数:         {len(failed)}")
    if failed:
        print(f"  拒识失败样例 (前5条):")
        for r in failed[:5]:
            print(f"    [{r['id']}] HYP: {r['hyp']}")

    return rr


def save_results(results, output_path):
    """保存推理结果到 JSONL 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[保存] 结果已保存到: {output_path}")


def load_results(result_path):
    """从 JSONL 文件加载已有结果"""
    if not result_path.exists():
        return None
    return load_jsonl(result_path)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR-1.7B 对 datasetA 推理与评估")
    parser.add_argument("--pos-only", action="store_true", help="仅推理 pos 集")
    parser.add_argument("--neg-only", action="store_true", help="仅推理 neg 集")
    parser.add_argument("--eval-only", action="store_true", help="仅评估 (使用已有结果文件)")
    parser.add_argument("--limit", type=int, default=0, help="限制推理条数 (测试用)")
    parser.add_argument("--cpu", action="store_true", help="使用 CPU 推理")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="模型名称或本地路径")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Qwen3-ASR-1.7B — datasetA 语音识别测试")
    print(f"{'='*60}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"  模型: {args.model}")
    print(f"  设备: {'CPU' if args.cpu else 'cuda:0'}")

    # ---- 仅评估模式 ----
    if args.eval_only:
        print("\n[模式] 仅评估 (使用已有结果文件)")
        pos_results = load_results(POS_RESULT)
        neg_results = load_results(NEG_RESULT)

        if pos_results:
            print(f"  加载 pos 结果: {len(pos_results)} 条")
            evaluate_pos(pos_results)
        else:
            print(f"  [警告] 未找到 pos 结果文件: {POS_RESULT}")

        if neg_results:
            print(f"  加载 neg 结果: {len(neg_results)} 条")
            evaluate_neg(neg_results)
        else:
            print(f"  [警告] 未找到 neg 结果文件: {NEG_RESULT}")
        return

    # ---- 加载模型 ----
    device = "cpu" if args.cpu else "cuda:0"
    model = load_model(args.model, device)

    limit = args.limit if args.limit > 0 else None

    # ---- POS 推理 ----
    if not args.neg_only:
        pos_samples = load_jsonl(POS_JSONL, limit=limit)
        print(f"\n[数据] 加载 pos 样本: {len(pos_samples)} 条")
        pos_results = run_inference_on_pos(model, pos_samples, DATASET_DIR)

        if pos_results:
            save_results(pos_results, POS_RESULT)
            evaluate_pos(pos_results)

    # ---- NEG 推理 ----
    if not args.pos_only:
        neg_samples = load_jsonl(NEG_JSONL, limit=limit)
        print(f"\n[数据] 加载 neg 样本: {len(neg_samples)} 条")
        neg_results = run_inference_on_neg(model, neg_samples, DATASET_DIR)

        if neg_results:
            save_results(neg_results, NEG_RESULT)
            evaluate_neg(neg_results)

    print(f"\n{'='*60}")
    print(f"  全部完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
