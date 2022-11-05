# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/4 9:01
@File: eval.py
@Desc: 
"""
import json
import os
from torch.utils.data import DataLoader
from transformers import BertForNextSentencePrediction, BertTokenizer, BertConfig

from config.config import Config
from dataloader import MyDataset, dataset_collect
from utils import setup_logger
from models import build_model

from train import test

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model_dir = './output/20221103-2152/checkpoint'
    model_name = 'SBert'
    config = Config()
    logger = setup_logger(os.path.join(config.output_dir, 'train.log'))
    print("Loading data...", flush=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    test_dataset = MyDataset(config.test_path, tokenizer, config)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=dataset_collect)
    print("Test...", flush=True)
    model = build_model(model_name, model_dir, best_name='best.ckpt')
    model.cuda()
    test(config, model, test_dataloader, logger)
