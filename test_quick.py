#!/usr/bin/env python3
"""
快速测试脚本 - 验证 Qwen3-ASR-1.7B 推理 pipeline
仅跑前3条，确认环境正确
"""
import sys
import json
import time
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

DATASET_DIR = Path("E:/ASR_Models/datasetA")
MODEL_DIR = Path("E:/qwen-asr/Qwen3-ASR-1.7B")

def main():
    print("=" * 50)
    print("  Qwen3-ASR-1.7B 快速测试")
    print("=" * 50)

    # 1. 检查模型
    if not (MODEL_DIR / "config.json").exists():
        print("[FAIL] 模型未下载完成！")
        return 1

    # 2. 加载模型
    print("\n[1/3] 加载模型...")
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"  设备: {device}, dtype: {dtype}")

    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        str(MODEL_DIR),
        dtype=dtype,
        device_map=device,
        max_new_tokens=256,
    )
    print(f"  模型加载耗时: {time.time() - t0:.1f}s")

    # 3. 加载测试数据
    print("\n[2/3] 加载测试数据...")
    pos_jsonl = DATASET_DIR / "pos.jsonl"
    with open(pos_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f.readlines()[:3] if line.strip()]

    print(f"  测试 {len(samples)} 条样本")

    # 4. 推理
    print("\n[3/3] 推理测试...")
    audio_paths = [str(DATASET_DIR / s["识别音频"]) for s in samples]

    for i, (sample, path) in enumerate(zip(samples, audio_paths)):
        if not Path(path).exists():
            print(f"  [{i}] [SKIP] 文件不存在: {path}")
            continue

        t1 = time.time()
        result = model.transcribe(audio=path, language="Chinese")
        elapsed = time.time() - t1

        ref = sample["识别文本"]
        hyp = result[0].text.strip() if result[0].text else ""

        # 快速 CER
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        n = len(ref_chars)
        m = len(hyp_chars)
        if n == 0 and m == 0:
            cer = 0.0
        elif n == 0:
            cer = 1.0
        else:
            dp = [[0]*(m+1) for _ in range(n+1)]
            for i in range(n+1):
                dp[i][0] = i
            for j in range(m+1):
                dp[0][j] = j
            for i in range(1, n+1):
                for j in range(1, m+1):
                    cost = 0 if ref_chars[i-1] == hyp_chars[j-1] else 1
                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
            cer = dp[n][m] / n

        print(f"  [{i}] REF: {ref}")
        print(f"  [{i}] HYP: {hyp}")
        print(f"  [{i}] CER: {cer*100:.2f}%, 耗时: {elapsed:.1f}s")
        print()

    print("=" * 50)
    print("  测试完成!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
