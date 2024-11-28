import sys
import torch
import yaml
import re
import json

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

def load_phones(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    phones_dict = {}
    for line in lines:
        wav_id, language, phones, text = line.strip().split('|')
        phones_dict[wav_id] = [phones, text]
    return phones_dict

def load_semantic(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    semantic_dict = {}
    for line in lines:
        wav_id, semantic = line.strip().split('|')
        semantic_dict[wav_id] = semantic
    return semantic_dict

def load_hparams(path):
    with open(path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    return hparams

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config

def get_newest_ckpt(string_list):
    # 定义一个正则表达式模式，用于匹配字符串中的数字
    pattern = r'epoch=(\d+)-step=(\d+)\.ckpt'

    # 使用正则表达式提取每个字符串中的数字信息，并创建一个包含元组的列表
    extracted_info = []
    for string in string_list:
        match = re.match(pattern, string)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            extracted_info.append((epoch, step, string))
    # 按照 epoch 后面的数字和 step 后面的数字进行排序
    sorted_info = sorted(
        extracted_info, key=lambda x: (x[0], x[1]), reverse=True)
    # 获取最新的 ckpt 文件名
    newest_ckpt = sorted_info[0][2]
    return newest_ckpt