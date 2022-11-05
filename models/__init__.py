# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/5 10:05
@File: __init__.py.py
@Desc: 
"""
import json
import os

import torch
from transformers import BertConfig

from .bert import Bert
from .sbert import SBert
from .losses import *


def build_model(model_name, pretrain_dir, best_name='best.ckpt'):
    model = eval(model_name)
    if (pretrain_dir and not os.path.exists(os.path.join(pretrain_dir, 'pytorch_model.bin'))) or (best_name != 'best.ckpt'):
        config_path = os.path.join(pretrain_dir, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        bert_config = BertConfig(**json_data)
        model = model(bert_config)
        model_path = os.path.join(pretrain_dir, best_name)
        print('load model: ', model_path)
        state_dict = torch.load(model_path)
        # 旧训练的模型
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('model.'):
                new_state_dict['model.' + k] = v
        model.load_state_dict(new_state_dict)
    elif pretrain_dir:
        print('load model: ', pretrain_dir)
        model = model.from_pretrained(pretrain_dir)
    else:
        print('load model: bert-base-chinese')
        model = model.from_pretrained('bert-base-chinese')
    return model

def build_loss():
    pass