#!/usr/bin/env python3
"""
环境检查脚本 - 验证 Qwen3-ASR-1.7B 环境是否正确搭建
"""
import os
import sys

# 修复 Windows GBK 编码问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasetA"

def check_header(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def check(label, ok, detail=""):
    status = "✓" if ok else "✗"
    print(f"  [{status}] {label}" + (f" - {detail}" if detail else ""))

def main():
    all_ok = True

    # 1. Python 环境
    check_header("Python 环境")
    check("Python 3.11+", sys.version_info >= (3, 11), f"Python {sys.version}")

    # 2. PyTorch + CUDA
    check_header("PyTorch & CUDA")
    try:
        import torch
        check("PyTorch 已安装", True, f"v{torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        check("CUDA 可用", cuda_ok)
        if cuda_ok:
            check("GPU 型号", True, torch.cuda.get_device_name(0))
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
            check("GPU 显存", vram_gb >= 4, f"{vram_gb:.1f} GB (≥4GB 推荐)")
        else:
            all_ok = False
    except ImportError:
        check("PyTorch 已安装", False, "未安装! pip install torch")
        all_ok = False

    # 3. qwen-asr
    check_header("qwen-asr 包")
    try:
        import qwen_asr
        check("qwen-asr 已安装", True, f"v{qwen_asr.__version__ if hasattr(qwen_asr, '__version__') else 'OK'}")
    except ImportError:
        check("qwen-asr 已安装", False, "未安装! pip install qwen-asr")
        all_ok = False

    # 4. transformers
    check_header("Transformers")
    try:
        import transformers
        check("transformers 已安装", True, f"v{transformers.__version__}")
    except ImportError:
        check("transformers 已安装", False)

    # 5. 模型文件
    check_header("模型文件")
    model_dirs = [
        BASE_DIR / "Qwen3-ASR-1.7B",
        BASE_DIR / "models" / "Qwen" / "Qwen3-ASR-1.7B",
    ]
    model_found = False
    for d in model_dirs:
        if d.exists():
            # 检查关键文件
            config = d / "config.json"
            if config.exists():
                check("模型已下载", True, str(d))
                model_found = True
                # 计算大小
                total_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                check("模型大小", True, f"{total_size/1024**3:.1f} GB")
                break

    if not model_found:
        check("模型已下载", False, "请先下载模型!")
        print(f"\n  下载命令:")
        print(f"    huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B")
        print(f"    或:")
        print(f"    modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./Qwen3-ASR-1.7B")
        all_ok = False

    # 6. 数据集
    check_header("数据集 (datasetA)")
    pos_file = DATASET_DIR / "pos.jsonl"
    neg_file = DATASET_DIR / "neg.jsonl"
    pos_dir = DATASET_DIR / "pos"
    neg_dir = DATASET_DIR / "neg"

    check("pos.jsonl", pos_file.exists(), f"{pos_file.stat().st_size} bytes" if pos_file.exists() else "")
    check("neg.jsonl", neg_file.exists(), f"{neg_file.stat().st_size} bytes" if neg_file.exists() else "")

    if pos_file.exists():
        import json
        with open(pos_file) as f:
            pos_count = sum(1 for _ in f)
        check("pos 样本数", True, f"{pos_count} 条")

    if neg_file.exists():
        import json
        with open(neg_file) as f:
            neg_count = sum(1 for _ in f)
        check("neg 样本数", True, f"{neg_count} 条")

    # 音频文件检查
    check_header("音频文件")
    pos_wav_count = len(list(pos_dir.glob("*.wav"))) if pos_dir.exists() else 0
    neg_wav_count = len(list(neg_dir.glob("*.wav"))) if neg_dir.exists() else 0

    check("pos 音频文件", pos_wav_count > 0, f"{pos_wav_count} 个 .wav 文件")
    check("neg 音频文件", neg_wav_count > 0, f"{neg_wav_count} 个 .wav 文件")

    if pos_wav_count == 0:
        print(f"\n  ⚠ 音频文件缺失!")
        print(f"  请将 pos 音频文件放入: {pos_dir}")
        print(f"  请将 neg 音频文件放入: {neg_dir}")
        print(f"  文件命名格式: cmd_0.wav, kws_0.wav (与 jsonl 中字段对应)")
        all_ok = False

    # 总结
    check_header("总结")
    if all_ok:
        print("  🎉 环境检查全部通过! 可以运行推理:")
        print(f"    python asr_inference.py")
    else:
        print("  ⚠ 存在问题需要解决，请检查上述 [✗] 项")

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
