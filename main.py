import utils.utils as utils
# import config
import logging
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from train import train
from train import test, translate
from data_loader import DatasetObj
from utils.utils import get_en_tokenizer
from model import Transformer, init_model_parameters
from labelSmoothing import LabelSmoothing
from utils.utils import get_hyperparameter, get_device

# 导入配置文件
args = get_hyperparameter()
# 获取运行环境
DEVICE = get_device(args)

d_model = args.d_model
d_ff = args.d_ff
n_heads =args.n_heads
dropout = args.dropout

warmup_steps = 10000
initial_lr = 0.1


def main():
    # 定义数据对象
    train_dataset = DatasetObj(args.train_zh_path, args.train_en_path)
    dev_dataset = DatasetObj(args.dev_zh_path, args.dev_en_path)
    test_dataset = DatasetObj(args.test_zh_path, args.test_en_path)

    print("加载数据...")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,collate_fn=train_dataset.code_sent)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,collate_fn=dev_dataset.code_sent)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.code_sent)

    print("构建模型...")
    model = Transformer(args.tgt_vocab_size, args.n_layers, d_model, d_ff, n_heads, dropout).to(DEVICE)
    # 模型参数初始化
    model = init_model_parameters(model)
    # 训练
    # 是否使用 标签平滑 定义代价
    if args.use_smoothing:
        criterion = LabelSmoothing(args.tgt_vocab_size, padding_idx=args.padding_idx, smoothing=0.1)
        criterion.to(DEVICE)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    # 是否使用 warm up 定义优化器
    if args.use_warmup:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (
                    d_model ** (-0.5) * min((epoch + 1) ** (-0.5), (epoch + 1) * warmup_steps ** (-1.5))))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optimizer
    print("开始训练...")
    # for data in train_dataloader:
    #     print(data.src)
    # print(train_dataloader)
    train(train_dataloader, dev_dataloader, model, criterion, scheduler)
    print("开始测试...")
    test(test_dataloader, model, criterion)

if __name__ == '__main__':
    main()


