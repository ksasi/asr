from thop import profile

import logging
import os
import sys
from pathlib import Path
import librosa
from taylor_series_linear_attention import TaylorSeriesLinearAttn

import torch
from torch import Tensor
from hyperpyyaml import load_hyperpyyaml
from inspect import signature 

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main

logger = logging.getLogger(__name__)
from speechbrain.inference.ASR import EncoderDecoderASR

conformer_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-conformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech")

input_path = '/home/ubuntu/asr/datasets/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac'
audio, org_sr = librosa.load(input_path, sr=16000)
audio_input = Tensor(audio).unsqueeze(0)

# MACs and Parameters for Vanilla Conformer

conformer_macs, conformer_params = profile(conformer_model, inputs=(audio_input, torch.tensor([1.0]), ))

print("Total MACs (Multiply-Accumulate Operations) of Conformer :", conformer_macs)
print("Total parameters of Conformer :", conformer_params)
print("\n")

# MACs and Parameters for TSConformer

TSConformer_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-conformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech")

# Define LinearAttention a wrapper for Taylor Series LinearAttention
class LinearAttn(torch.nn.Module):
    """Wrapper Class for Taylor Series LinearAttention"""
    def __init__(self, dim = 512, dim_head = 16, heads = 8):
        super(LinearAttn, self).__init__()
        self.attn = TaylorSeriesLinearAttn(dim = dim, dim_head = dim_head, heads = heads)

    def forward(self, query, key, value, attn_mask, key_padding_mask, pos_embs):
        # Ignoring key_padding_mask and pos_embs for TaylorSeriesLinearAttn
        out = self.attn(query, mask=attn_mask)
        return out, self.attn

for i in range(12):
    TSConformer_model.mods.transformer.encoder.layers[i].mha_layer = LinearAttn(dim = 512, dim_head = 16, heads = 8).cuda()
    TSConformer_model.mods.transformer.encoder.layers[i].mha_layer.requires_grad = True
    TSConformer_model.mods.transformer.encoder.layers[i].mha_layer.attn.requires_grad = True


tsconformer_macs, tsconformer_params = profile(conformer_model, inputs=(audio_input, torch.tensor([1.0]), ))

print("Total MACs (Multiply-Accumulate Operations) of TSConformer :", tsconformer_macs)
print("Total parameters of TSConformer :", tsconformer_params)


