# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/3 19:04
@File: data_statics.py
@Desc: 
"""
import pandas as pd
import matplotlib
from pylab import mpl
import matplotlib.pyplot as plt
from collections import Counter

def data_statics(file_path):
    if isinstance(file_path, str):
        file_path = [file_path]
    data = []
    data_len = []
    for file in file_path:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            texta, textb, label = line.split('\t')
            data.append(texta)
            data.append(textb)
            data_len.append(len(texta))
            data_len.append(len(textb))
    print('total: {}, ave len: {}, max len: {}, min len:{}'.format(len(data_len), sum(data_len)/len(data_len), max(data_len), min(data_len)))
    len_counter = Counter(data_len)
    plt.bar(list(len_counter.keys()), list(len_counter.values()))
    plt.xlabel('text len')
    plt.ylabel('num')
    # plt.show()
    plt.savefig('./output/data.png')


def data_check(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    data = []
    for line in lines:
        if len(line.strip().split('\t')) != 3:
            continue
        data.append(line.strip())
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))


if __name__ == '__main__':
    file_path = [
        '/data/wuzhichao/homework/text_sim/data/paws-x/train.tsv',
        '/data/wuzhichao/homework/text_sim/data/paws-x/dev.tsv',
        '/data/wuzhichao/homework/text_sim/data/bq_corpus/train.tsv',
        '/data/wuzhichao/homework/text_sim/data/bq_corpus/dev.tsv',
        '/data/wuzhichao/homework/text_sim/data/lcqmc/train.tsv',
        '/data/wuzhichao/homework/text_sim/data/lcqmc/dev.tsv'
    ]
    data_statics(file_path)
    # data_check(file_path)
