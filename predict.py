# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/3 21:11
@File: predict.py
@Desc: 
"""
import json
import os
import time

import pandas as pd
from tqdm import tqdm
import torch
import pickle as pkl
from transformers import BertForNextSentencePrediction, BertTokenizer, BertConfig
from models import build_model
import transformers

transformers.logging.set_verbosity_error()


class Predict:
    def __init__(self, model_name, model_dir, best_name):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = build_model(model_name, model_dir, best_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = 100

    def __call__(self, texta, textb):
        encoding1 = self.tokenizer(texta, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding2 = self.tokenizer(textb, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding = self.tokenizer(texta, textb, add_special_tokens=True, max_length=self.max_length*2, padding='max_length', truncation=True, return_tensors='pt')
        encoding1 = {k: v.to(self.device) for k, v in encoding1.items()}
        encoding2 = {k: v.to(self.device) for k, v in encoding2.items()}
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        outputs = self.model([encoding1, encoding2, encoding])
        predict = torch.max(outputs, 1)[1].cpu().numpy().tolist()
        return predict


def test(file_path, model, dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pred = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        texta, textb = line.strip().split('\t')
        output = model(texta, textb)
        pred.append(output[0])
    df = pd.DataFrame(pred)
    df.to_csv(os.path.join(save_dir, f'{dataset}.tsv'), sep='\t', index_label='index', header=['prediction'])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_path = './output/SBert-20221105-1555/checkpoint'
    model_name = 'SBert'
    model = Predict(model_name, model_path, best_name='best.ckpt')
    test_file = [
        '/data/wuzhichao/homework/text_sim/data/bq_corpus/test.tsv',
        '/data/wuzhichao/homework/text_sim/data/lcqmc/test.tsv',
        '/data/wuzhichao/homework/text_sim/data/paws-x/test.tsv'
    ]
    # save_dir = f'./output/res_{time.strftime("%Y%m%d_%H")}'
    save_dir = f'./output/res_20221105_13'
    for file in test_file:
        dataset = os.path.split(os.path.split(file)[0])[-1]
        print(dataset)
        test(file, model, dataset, save_dir)
