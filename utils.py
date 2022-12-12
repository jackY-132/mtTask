from tqdm import tqdm
import os
import logging
import sentencepiece as spm
import yaml
import numpy as np
from attrdict import AttrDict
import torch
def get_ch_tokenizer():
    sp_chn = spm.SentencePieceProcessor()
    sp_chn.Load("/tmp/nlp/mtdata/80000/chinese.model")
    return sp_chn


def get_en_tokenizer():
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load("/tmp/nlp/mtdata/80000/english.model")
    return sp_eng

def get_hyperparameter():
    # 导入配置文件
    yaml_file = '/tmp/nlp/work/transformer.config.yaml'
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
    args.lr = float(args.lr)
    return args

def get_device(args):
    if args.use_gpu and torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device('cpu')
    return DEVICE

# 生成一个三角矩阵，用于盖住decoder的输入
def get_future_mask(size):
    """Mask out subsequent positions."""
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)

    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


def filter_out_html(filename1, filename2):
    f1 = open(filename1, 'r')
    f2 = open(filename2, 'r')

    data1 = f1.readlines()
    data2 = f2.readlines()
    assert len(data1) == len(data2)  # 用codecs会导致报错不知道为什么
    fw1 = open(filename1 + ".txt", 'w')
    fw2 = open(filename2 + ".txt", 'w')

    for line1, line2 in tqdm(zip(data1, data2)):
        line1 = line1.strip()
        line2 = line2.strip()
        if line1 and line2:
            if '<' not in line1 and '>' not in line1 and '<' not in line2 and '>' not in line2:
                fw1.write(line1 + "\n")
                fw2.write(line2 + "\n")
    fw1.close()
    f1.close()
    fw2.close()
    f2.close()
    print("结束！！")

    return filename1 + ".txt", filename2 + ".txt"


if __name__ == '__main__':
    # sp_chn = chinese_tokenizer_load()
    # print(sp_chn.data)
    filter_out_html("/tmp/nlp/mtdata/train.tags.zh-en.en", "/tmp/nlp/mtdata/train.tags.zh-en.zh")
