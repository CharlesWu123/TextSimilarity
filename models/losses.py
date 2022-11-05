# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/5 15:27
@File: losses.py
@Desc: 
"""
import torch.nn.functional as F


def cross_entropy(outputs, labels):
    return F.cross_entropy(outputs, labels)


def triple_loss(outputs, triplet_margin=5):
    rep_anchor, rep_pos, rep_neg = outputs
    distance_pos = F.pairwise_distance(rep_anchor, rep_pos)
    distance_neg = F.pairwise_distance(rep_anchor, rep_neg)
    losses = F.relu(distance_pos - distance_neg + triplet_margin)
    return losses.mean()