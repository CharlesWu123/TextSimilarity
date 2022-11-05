# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/3 17:11
@File: config.py
@Desc: 
"""
import os
import time


class Config(object):
    """配置参数"""

    def __init__(self, train=True):
        self.model_name = 'SBert'   # Bert, SBert
        time_str = time.strftime("%Y%m%d-%H%M")
        output_dir = f'./output/{self.model_name}-{time_str}'
        if train:
            os.makedirs(output_dir, exist_ok=True)

        self.train_path = [
            './data/bq_corpus/train.tsv',
            './data/lcqmc/train.tsv',
            './data/paws-x/train.tsv',
        ]  # 训练集
        self.test_path = [
            './data/bq_corpus/dev.tsv',
            './data/lcqmc/dev.tsv',
            './data/paws-x/dev.tsv'
        ]  # 测试集
        self.output_dir = output_dir

        self.require_improvement = 5000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 3  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.max_length = 80  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.pretrain_dir = '/data/wuzhichao/homework/text_sim/output/SBert-20221105-1242/checkpoint'

        self.log_iter = 100
        self.warmup = True
        self.warmup_epoch = 1
