# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/3 15:20
@File: dataloader.py
@Desc: 
"""
import os
import json
import torch
from torch.utils.data import Dataset


def dataset_collect(batch):
    new_batch = [
        {'input_ids': [], 'token_type_ids': [], 'attention_mask': []},
        {'input_ids': [], 'token_type_ids': [], 'attention_mask': []},
        {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
    ]
    labels = []
    for encodings, label in batch:
        for idx, encoding in enumerate(encodings):
            for k, v in encoding.items():
                new_batch[idx][k].append(v)
        labels.append(label)
    for batch in new_batch:
        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)
    labels = torch.LongTensor(labels)
    return new_batch, labels


class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, config):
        self.data = []
        self.tokenizer = tokenizer
        self.config = config
        if isinstance(file_path, str):
            file_path = [file_path]
        for file in file_path:
            self.load_data(file)

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            texta, textb, label = line.strip().split('\t')
            self.data.append((texta, textb, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        texta, textb, label = self.data[item]
        encoding1 = self.tokenizer(texta, add_special_tokens=True, max_length=self.config.max_length, padding='max_length', truncation=True)
        encoding2 = self.tokenizer(textb, add_special_tokens=True, max_length=self.config.max_length, padding='max_length', truncation=True)
        encoding = self.tokenizer(texta, textb, add_special_tokens=True, max_length=self.config.max_length, padding='max_length', truncation=True)
        return (encoding1, encoding2, encoding), label


