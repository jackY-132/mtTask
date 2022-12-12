import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.utils import get_ch_tokenizer
from utils.utils import get_en_tokenizer, get_hyperparameter, get_device, get_future_mask
from torch.utils.data import DataLoader
import sentencepiece as spm

# 导入配置文件
args = get_hyperparameter()
# 获取运行环境
DEVICE = get_device(args)

class DatasetObj(Dataset):
    def __init__(self, ch_data_path, en_data_path):
        # 中英文原始句子
        self.out_en_sent, self.out_cn_sent = self.get_dataset(ch_data_path, en_data_path, sort=True)
        # 切好的句子
        self.sp_eng = get_en_tokenizer()
        self.sp_chn = get_ch_tokenizer()
        # 一些特殊符号的标号
        self.PAD = self.sp_eng.pad_id()  # 0
        self.BOS = self.sp_eng.bos_id()  # 2
        self.EOS = self.sp_eng.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, ch_data_path, en_data_path, sort=False):
        out_en_sent = []
        out_cn_sent = []
        with open(ch_data_path, 'r') as f:
            for ids, row in enumerate(f.readlines()):
                out_cn_sent.append(row.strip())
        with open(en_data_path, 'r') as f:
            for ids, row in enumerate(f.readlines()):
                out_en_sent.append(row.strip())
        if sort:
            sorted_index = self.len_argsort(out_en_sent) # 按照英文句子的长度进行排序
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        return len(self.out_en_sent)

    # 把句子转成编号，并加上开始和结束标志
    def code_sent(self, batch):
        src_text = [x[0] for x in batch] # 所有的英语句子
        tgt_text = [x[1] for x in batch] # 所有的中文句子
        # 将字符串 编码为 id， 并加上开始和结束标志
        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]
        # 转化为 tensor 并 填充 0
        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens], batch_first=True, padding_value=0)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens], batch_first=True, padding_value=0)
        return Batch(src_text, tgt_text, batch_input, batch_target, 0)

class Batch:
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text # 所有的英文句子
        self.trg_text = trg_text # 所有的中文句子
        src = src.to(DEVICE)
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            trg = trg.to(DEVICE)
            self.trg = trg[:, :-1] # 编码器的输入
            self.trg_y = trg[:, 1:] # 对应的正确输出
            self.trg_mask = self.make_std_mask(self.trg, pad) # decoder输入编码
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    def make_std_mask(self, tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(get_future_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

if __name__ == '__main__':
    sp = spm.SentencePieceProcessor()
    sp.Load("/tmp/nlp/mtdata/chn.model")
    # train_dataset = MTDataset("/tmp/nlp/mtdata/two/train.json")
    train_dataset = DatasetObj("/tmp/nlp/mtdata/train.tags.zh-en.zh.txt","/tmp/nlp/mtdata/train.tags.zh-en.en.txt")
    # print(train_dataset.out_en_sent[-1000:])
    # print(train_dataset.out_cn_sent[-1000:])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,collate_fn=train_dataset.collate_fn)
    for ids, batch in enumerate(train_dataloader):
        print(batch.src)   # 英文
        print(batch.trg)   # 中文
        print(batch.src.shape)  # [32, 80] [batch_size, sent_len]
        print(batch.trg.shape)  # [32, 41] [batch_size, sent_len]
        print(batch.trg[0])
        print(sp.decode_ids(batch.trg[0].tolist()))
        break
