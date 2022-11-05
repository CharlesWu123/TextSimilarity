# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/5 10:05
@File: bert.py
@Desc: 
"""
import json
import os

import torch
import torch.nn as nn
from transformers import BertForNextSentencePrediction, BertTokenizer, BertConfig, BertPreTrainedModel


class Bert(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert, self).__init__(config)
        model = BertForNextSentencePrediction(config)
        self.model = model

    def forward(self, x):
        outputs = self.model(**x[2], return_dict=True)
        return outputs.logits
        