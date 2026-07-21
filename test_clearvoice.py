#!/usr/bin/env python3
"""Test ClearVoice speech separation with HF mirror"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from clearvoice import ClearVoice
import soundfile as sf
import numpy as np
import tempfile

print('Loading ClearVoice speech separation...')
cv = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
print('Loaded!')

cmd, sr = sf.read('E:/ASR_Models/datasetA/pos/cmd_0.wav')
print(f'Input: {len(cmd)/sr:.1f}s, {sr}Hz')

print('Separating...')
output = cv(input_path='E:/ASR_Models/datasetA/pos/cmd_0.wav', online_write=False)
print(f'Separated into {len(output)} streams')
for i, s in enumerate(output):
    print(f'  Stream {i}: shape={s.shape}, dur={s.shape[-1]/sr:.1f}s')
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir='E:/qwen-asr/tmp')
    audio = s.T if s.shape[0] < s.shape[1] else s
    sf.write(tmp.name, audio, sr)
    print(f'    saved to {tmp.name}')
    tmp.close()
