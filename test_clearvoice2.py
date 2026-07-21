#!/usr/bin/env python3
"""Test: ClearVoice separation → speaker matching → ASR"""
import os, sys, io, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from clearvoice import ClearVoice
from funasr import AutoModel
from qwen_asr import Qwen3ASRModel
import soundfile as sf
import numpy as np
import torch

# Load models
print('Loading ClearVoice...')
cv = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

print('Loading CAMPPlus...')
spk_model = AutoModel(model='iic/speech_campplus_sv_zh-cn_16k-common', model_revision='v2.0.2', disable_update=True)

print('Loading Qwen3-ASR...')
asr_model = Qwen3ASRModel.from_pretrained(
    'E:/qwen-asr/Qwen3-ASR-1.7B', dtype=torch.bfloat16, device_map='cuda:0', max_new_tokens=256,
)

def get_emb(model, path):
    r = model.generate(input=path)
    return r[0]['spk_embedding'].cpu().numpy().flatten()

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Test on a sample that had poor ASR before
# Let's test sample 0 and a known bad sample
for sid in [0, 50, 100]:
    cmd_path = f'E:/ASR_Models/datasetA/pos/cmd_{sid}.wav'
    kws_path = f'E:/ASR_Models/datasetA/pos/kws_{sid}.wav'

    cmd, sr = sf.read(cmd_path)
    kws, sr_k = sf.read(kws_path)

    print(f'\n=== Sample {sid} ({len(cmd)/sr:.1f}s) ===')

    # 1. ClearVoice separation
    streams = cv(input_path=cmd_path, online_write=False)

    # 2. Match streams to kws speaker
    kws_emb = get_emb(spk_model, kws_path)

    for i, stream in enumerate(streams):
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio = stream.squeeze() if hasattr(stream, 'squeeze') else np.squeeze(stream)
        if hasattr(audio, 'cpu'): audio = audio.cpu().numpy()
        sf.write(tmp.name, audio, sr)
        tmp.close()

        emb = get_emb(spk_model, tmp.name)
        sim = cosine(kws_emb, emb)

        # ASR on this stream
        r = asr_model.transcribe(audio=tmp.name, language='Chinese')
        hyp = r[0].text.strip()

        print(f'  Stream {i}: cos_sim={sim:.3f} | {hyp[:60]}')
        os.unlink(tmp.name)

    # Baseline: direct ASR on original audio
    r = asr_model.transcribe(audio=cmd_path, language='Chinese')
    print(f'  Original:  {"(no separation)"} | {r[0].text.strip()[:60]}')
