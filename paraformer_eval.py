#!/usr/bin/env python3
"""Paraformer vs Qwen3-ASR CER 对比"""
import json, re, sys, io, time
from pathlib import Path
from funasr import AutoModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATASET_DIR = Path("E:/ASR_Models/datasetA")
PUNCT = re.compile(r'[，。！？、；：""''（）《》【】…—·,\.!\?;:\"\'\(\)\[\]{}<>\s]')

def strip(text):
    return PUNCT.sub("", text)

def compute_cer(ref, hyp):
    # strip spaces (Paraformer adds them) + punctuation
    ref_c = list(strip(ref).replace(" ", ""))
    hyp_c = list(strip(hyp).replace(" ", ""))
    n, m = len(ref_c), len(hyp_c)
    if n == 0 and m == 0: return 0.0
    if n == 0: return float(m)
    dp = [[0]*(m+1) for _ in range(2)]
    dp[0] = list(range(m+1))
    for i in range(1, n+1):
        cur, prev = i%2, 1-i%2
        dp[cur][0] = i
        for j in range(1, m+1):
            cost = 0 if ref_c[i-1] == hyp_c[j-1] else 1
            dp[cur][j] = min(dp[prev][j]+1, dp[cur][j-1]+1, dp[prev][j-1]+cost)
    return dp[n%2][m] / n

print("Loading Paraformer...")
model = AutoModel(
    model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    disable_update=True,
)
print("Loaded!")

with open(DATASET_DIR / "pos.jsonl", encoding="utf-8") as f:
    samples = [json.loads(l) for l in f if l.strip()]

# Load Qwen3 baseline
with open("E:/qwen-asr/pos_result.jsonl", encoding="utf-8") as f:
    qwen_results = {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}

results = []
t0 = time.time()

for idx, s in enumerate(samples):
    sid = s["id"]
    ref = s["识别文本"]
    path = str(DATASET_DIR / s["识别音频"])

    try:
        r = model.generate(input=path)
        hyp = r[0]["text"].strip()
        cer = compute_cer(ref, hyp)
    except Exception:
        hyp, cer = "[ERROR]", 1.0

    qwen_cer = qwen_results.get(sid, {}).get("cer", 1.0)
    results.append({"id": sid, "ref": ref, "hyp": hyp, "cer": cer, "qwen_cer": qwen_cer})

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        speed = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (len(samples) - idx - 1) / speed if speed > 0 else 0
        print(f"  {idx+1}/{len(samples)} | {speed:.1f}/s | ETA {eta/60:.1f}min")

# Evaluation
cers = [r["cer"] for r in results]
qwen_cers = [r["qwen_cer"] for r in results]
avg = sum(cers) / len(cers)
avg_qwen = sum(qwen_cers) / len(qwen_cers)
improved = sum(1 for r in results if r["cer"] < r["qwen_cer"])
worsened = sum(1 for r in results if r["cer"] > r["qwen_cer"])

print(f"\n{'='*55}")
print(f"  Paraformer vs Qwen3-ASR")
print(f"{'='*55}")
print(f"  Paraformer CER:    {avg*100:.2f}%")
print(f"  Qwen3-ASR CER:     {avg_qwen*100:.2f}%")
print(f"  改善: {improved}, 变差: {worsened}")

# By ID range
for label, lo, hi in [("Low  <2000", 0, 2000), ("High >=2000", 2000, 9999)]:
    g = [r for r in results if lo <= r["id"] < hi]
    pf = sum(r["cer"] for r in g) / len(g) if g else 0
    qw = sum(r["qwen_cer"] for r in g) / len(g) if g else 0
    print(f"  {label}: PF {pf*100:.1f}% vs Qwen {qw*100:.1f}%")

with open("E:/qwen-asr/pos_paraformer_result.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nSaved to pos_paraformer_result.jsonl")
