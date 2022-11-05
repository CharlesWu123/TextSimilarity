# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/5 10:12
@File: sbert.py
@Desc: 
"""
import json
import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig, BertPreTrainedModel
from sentence_transformers import SentenceTransformer


class SBert(BertPreTrainedModel):
    def __init__(self, config):
        super(SBert, self).__init__(config)
        self.bert = BertModel(config)
        self.fc = nn.Linear(3*self.bert.config.hidden_size, 2)

    def mean_pool(self, token_embeddings, attention_mask):
        # input_mask_expanded = self.bert.get_extended_attention_mask(attention_mask, token_embeddings.size())
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, x):
        outputs1 = self.bert(**x[0], return_dict=True)
        outputs2 = self.bert(**x[1], return_dict=True)
        # pool out
        # pool1 = outputs1.pooler_output
        # pool2 = outputs2.pooler_output
        # mean
        pool1 = self.mean_pool(outputs1.last_hidden_state, x[0]['attention_mask'])
        pool2 = self.mean_pool(outputs2.last_hidden_state, x[1]['attention_mask'])

        sentence = torch.cat((pool1, pool2, torch.abs(pool1 - pool2)), 1)
        logits = self.fc(sentence)
        return logits
