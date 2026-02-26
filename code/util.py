import numpy as np
import random
import os
import torch

def clean(tok):
    return tok[2:] if tok.startswith("##") else tok
def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def print_config(args, file_id):
    """
    把 args 里的所有参数按字母顺序写进 config.txt。
    如果中间目录不存在就自动创建。
    """
    # 拼出目标目录和文件路径
    target_dir = os.path.join(args.base_path, args.dataset, "results", file_id)
    os.makedirs(target_dir, exist_ok=True)          # 目录不存在就创建
    config_path = os.path.join(target_dir, "config.txt")

    # 写入配置
    with open(config_path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            print(k, '=', v, file=f)
def mat_padding(inputs, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]
    if length is None:
        length = max([x.shape[0] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[0] = (0, length - x.shape[0])
        pad_width[1] = (0, length - x.shape[0])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)


def matr_padding(inputs, length=None, padding=0):

    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]
    if length is None:
        length0 = max([x.shape[0] for x in inputs])
        length1 = max([x.shape[1] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    
    for x in inputs:
        pad_width[0] = (0, length0 - x.shape[0])
        pad_width[1] = (0, length1 - x.shape[1])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

def sequence_padding(inputs, dim=0, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]
    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[dim] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)
import unicodedata

def strip_accents(dataset,text):
    """去除重音符号（简化版本）"""
    if dataset == "NYT" or dataset == "NYT_star":
        try:
            return ''.join(c for c in unicodedata.normalize('NFD', text) 
                        if unicodedata.category(c) != 'Mn')
        except:
            # 如果unicodedata不可用，返回原文本
            return text
    else:
        return text