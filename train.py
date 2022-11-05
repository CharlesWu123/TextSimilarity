# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/3 14:55
@File: train.py
@Desc: 
"""
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, WarmupPolyLR, setup_logger
from tensorboardX import SummaryWriter
from dataloader import MyDataset, dataset_collect
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader
import transformers
from models import build_model


from config.config import Config

transformers.logging.set_verbosity_error()


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    writer = SummaryWriter(log_dir=config.output_dir)
    logger = setup_logger(os.path.join(config.output_dir, 'train.log'))
    model_save_dir = os.path.join(config.output_dir, 'checkpoint')
    os.makedirs(model_save_dir, exist_ok=True)
    # 训练配置
    logger.info(json.dumps(vars(config), ensure_ascii=False, indent=2))
    with open(os.path.join(config.output_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(vars(config), ensure_ascii=False, indent=2))
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_dataset = MyDataset(config.train_path, tokenizer, config)
    test_dataset = MyDataset(config.test_path, tokenizer, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=dataset_collect)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=dataset_collect)
    # 保存vocab.txt
    tokenizer.save_pretrained(model_save_dir)
    model = build_model(config.model_name, config.pretrain_dir)
    model.to(device)
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.warmup:
        warmup_iters = config.warmup_epoch * len(train_dataloader)
        scheduler = WarmupPolyLR(optimizer, max_iters=config.num_epochs * len(train_dataloader), warmup_iters=warmup_iters,
                                 warmup_epoch=config.warmup_epoch, last_epoch=-1)
    total_batch = 0  # 记录进行到多少batch
    test_best_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    logger.info('train dataset has {} samples, {} in dataloader, validate dataset has {} samples, {} in dataloader'.
                format(len(train_dataloader.dataset), len(train_dataloader), len(test_dataloader.dataset), len(test_dataloader)))
    for epoch in range(config.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        for i, batch in enumerate(train_dataloader):
            lr = optimizer.param_groups[0]['lr']
            encodings, labels = batch
            for idx, encoding in enumerate(encodings):
                encodings[idx] = {k: v.to(device) for k, v in encoding.items()}
            labels = labels.to(device)
            outputs = model(encodings)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if config.warmup:
                scheduler.step()
            if total_batch % config.log_iter == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predict = torch.max(outputs, 1)[1].cpu()
                # train_acc = metrics.accuracy_score(true, predict)
                train_acc, train_rec, train_f1, _ = metrics.precision_recall_fscore_support(true, predict,
                                                                                            average='macro',
                                                                                            zero_division=0)
                test_acc, test_rec, test_f1, test_loss = evaluate(config, model, test_dataloader)
                if test_acc > test_best_acc:
                    test_best_acc = test_acc
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f'best_{epoch+1}.ckpt'))
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f'best.ckpt'))
                    model.save_pretrained(model_save_dir)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Epoch [{}/{}], Iter: {:>6},  Train Loss: {:>5.2f},  Train Acc: {:>7.2%},  Test Loss: {:>5.2f}, Test Acc: {:>6.2%}, LR: {:>7.6f},Time: {} {}'
                logger.info(msg.format(epoch + 1, config.num_epochs, total_batch, loss.item(), train_acc, test_loss, test_acc, lr, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/test", test_loss, total_batch)
                writer.add_scalar("f1/train", train_f1, total_batch)
                writer.add_scalar("f1/test", test_f1, total_batch)
                writer.add_scalar("train/lr", lr, total_batch)

                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_dataloader, logger, os.path.join(model_save_dir, f'best.ckpt'))


def test(config, model, test_iter, logger, best_model_path=''):
    # test
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
    model.eval()
    start_time = time.time()
    test_acc, test_recall, test_f1, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Rec: {2:>6.2%}, Test F1: {3:>6.2%}'
    logger.info(msg.format(test_loss, test_acc, test_recall, test_f1))
    logger.info("\nPrecision, Recall and F1-Score...")
    logger.info('\n{}'.format(test_report))
    logger.info("Confusion Matrix...\n")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    logger.info("Time usage:".format(time_dif))


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            encodings, labels = batch
            for idx, encoding in enumerate(encodings):
                encodings[idx] = {k: v.cuda() for k, v in encoding.items()}
            labels = labels.cuda()
            outputs = model(encodings)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    # acc = metrics.accuracy_score(labels_all, predict_all)
    acc, recall, f1, _ = metrics.precision_recall_fscore_support(labels_all, predict_all, average='macro', zero_division=0)
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, recall, f1, loss_total / len(data_iter), report, confusion
    return acc, recall, f1, loss_total / len(data_iter)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = Config()
    train(config)