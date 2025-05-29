import argparse
import copy
import os

import torch
import torch.nn as nn
from typing import Optional
from datautils import *
from modelutils import *
from quant import *
from trueobs import *
from datasets import load_dataset
from transformers import AutoImageProcessor
from torch.nn import Linear, Conv2d
import ipdb



def _calibrate_input(m: torch.nn.Module, input: torch.Tensor, momentum: float = 0.9):
    input = input[0]    
    new_scale = input.abs().max()/m.maxq
    if not hasattr(m,'act_scale'):
        m.register_buffer('act_scale',new_scale)
    else:
        m.act_scale = momentum * m.act_scale + new_scale * (1.0 - momentum)
    return input

def _quantize_input(m: torch.nn.Module, input: torch.Tensor):
    input = input[0]    
    qdq_input = torch.clamp(torch.round(input / m.act_scale) , m.minq, m.maxq)*m.act_scale
    return qdq_input

def get_module(module: nn.Module, name: str):
    """
    점(.)으로 구분된 이름 문자열을 사용하여 모듈을 가져옵니다.
    예: 'layer3.1.conv2'
    """
    names = name.split('.')
    current_module = module
    for n_part in names:
        # hasattr를 사용하여 안전하게 접근하거나, 바로 getattr 사용
        if hasattr(current_module, n_part):
            current_module = getattr(current_module, n_part)
        else:
            # nn.ModuleDict, nn.Sequential, nn.ModuleList의 경우 숫자로 된 키/인덱스 처리
            try:
                # 숫자로 된 이름 부분 (예: '0', '1')을 처리
                # ModuleDict는 문자열 키를 사용하므로, getattr로 처리됨
                # Sequential, ModuleList는 getattr(module, '0') 처럼 접근 가능하거나,
                # current_module = current_module[n_part] (ModuleDict)
                # current_module = current_module[int(n_part)] (Sequential, ModuleList)
                # 하지만 named_modules()가 반환하는 이름은 getattr로 대부분 접근 가능합니다.
                # 만약 n_part가 숫자로만 구성되어 있고 getattr로 실패하면, 인덱싱 시도
                if n_part.isdigit() and not isinstance(current_module, nn.ModuleDict):
                    current_module = current_module[int(n_part)]
                else: # ModuleDict의 경우 문자열 키 그대로 사용
                    current_module = current_module[n_part]

            except (KeyError, IndexError, TypeError) as e:
                raise AttributeError(f"Module {type(current_module).__name__} has no attribute or key {n_part} in name {name}") from e
    return current_module

def transform(data_batch, processor):
    IMAGE = data_batch["image"]
    IMAGE = [image.convert('RGB') for image in IMAGE]
    inputs = processor(IMAGE, return_tensors="pt")
    inputs["labels"] = data_batch["label"]
    return inputs

parser = argparse.ArgumentParser()

parser.add_argument('model', type=str)
parser.add_argument('dataset', type=str)
parser.add_argument(
    'compress', type=str, choices=['quant', 'nmprune', 'unstr', 'struct', 'blocked']
)
parser.add_argument('--load', type=str, default='')
parser.add_argument('--datapath', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save', type=str, default='')

parser.add_argument('--nsamples', type=int, default=1024)
parser.add_argument('--batchsize', type=int, default=-1)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--nrounds', type=int, default=-1)
parser.add_argument('--noaug', action='store_true')

parser.add_argument('--wbits', type=int, default=32)
parser.add_argument('--abits', type=int, default=32)
parser.add_argument('--wperweight', action='store_true')
parser.add_argument('--wasym', action='store_true')
parser.add_argument('--wminmax', action='store_true')
parser.add_argument('--asym', action='store_true')
parser.add_argument('--aminmax', action='store_true')
parser.add_argument('--rel-damp', type=float, default=0)

parser.add_argument('--prunen', type=int, default=2)
parser.add_argument('--prunem', type=int, default=4)
parser.add_argument('--blocked_size', type=int, default=4)
parser.add_argument('--min-sparsity', type=float, default=0)
parser.add_argument('--max-sparsity', type=float, default=0)
parser.add_argument('--delta-sparse', type=float, default=0)
parser.add_argument('--sparse-dir', type=str, default='')

args = parser.parse_args()

ds = load_dataset(path="Tsomaros/Imagenet-1k_validation", cache_dir='/Data/Dataset/ImageNet', split='validation')
processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=256, shuffle=True, num_workers=8)


get_model, test, run = get_functions(args.model)
model = get_model()

trueobs = {}
for name, m in model.named_modules():
    if not isinstance(m,(Linear,Conv2d)): continue
    trueobs[name] = TrueOBS(m, rel_damp=args.rel_damp)
    trueobs[name].quantizer = Quantizer()
    trueobs[name].quantizer.configure(
        args.wbits, perchannel=True, sym=True, mse=False
    )



def add_batch(name):
    def tmp(layer, inp, out):
        trueobs[name].add_batch(inp[0].data, out.data)
    return tmp
handles = []
for name, m in model.named_modules():
    if not isinstance(m,(Linear,Conv2d)): continue
    handles.append(m.register_forward_hook(add_batch(name)))

for i in range(1): # 간이 테스트
    for j, batch in enumerate(dataloader):
        print(i, j)
        with torch.no_grad():
            run(model, batch)
        if j==4 : break # 간이 테스트
for h in handles:
    h.remove()    


# W quantization
for name in trueobs:
    print(name)
    print('Quantizing ...')
    trueobs[name].quantize()
    m = get_module(model,name)
    m.register_buffer('w_scale',trueobs[name].quantizer.scale)
    trueobs[name].free()
  




# Act quantization
handles = []
for name, m in model.named_modules():
    if not isinstance(m,(Linear,Conv2d)): continue
    m.register_buffer('maxq',torch.tensor(2**(args.wbits-1)-1, device=m.weight.device))
    m.register_buffer('minq',torch.tensor(-2**(args.wbits-1),  device=m.weight.device))
    handles.append(m.register_forward_pre_hook(_calibrate_input))   
for j, batch in enumerate(dataloader):
    with torch.no_grad():
        run(model, batch)
    if j ==4 : break
for h in handles:
    h.remove() 


handles = []
for name, m in model.named_modules():
    if not isinstance(m,(Linear,Conv2d)): continue
    m.register_forward_pre_hook(_quantize_input)


# 평가
for j, batch in enumerate(dataloader):
    with torch.no_grad():
        run(model, batch)
    if j ==4 : break
