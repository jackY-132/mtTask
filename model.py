import numpy as np
import math
import copy
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_hyperparameter, get_device
from data_loader import get_future_mask

# 导入配置文件
args = get_hyperparameter()
# 获取运行环境
DEVICE = get_device(args)

# 进行位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0) # 将pe矩阵以持久的buffer状态存下
        self.register_buffer('pe', pos_encoding)
        # print("pe.shape: ", pe.shape) # [1, 100, 512]
        self.pos_encoding = pos_encoding

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        # 输入的是已经embedding编码后的数据，输出的是在输入的基础上加上位置编码的结果
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 生成注意力矩阵
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, Q, K, V, mask=None, dropout=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v, d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if mask != None:
            scores.masked_fill_(mask, -1e9)  # 为mask中为True的值填充-1e9。
        attn = nn.Softmax(dim=-1)(scores)
        if dropout is not None:
            attn = dropout(attn)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn  # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.d_k = d_model // h # 得到一个head的attention表示维度
        self.h = h # 头数
        self.d_model = d_model
        self.L_Q = nn.Linear(d_model, d_model, bias=False)
        self.L_K = nn.Linear(d_model, d_model, bias=False)
        self.L_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        residual = query # # 保存输入 用作残差
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_b = query.size(0)  # Q: [batch_size, n_heads, len_q, d_k]
        # 将embedding层乘以WQ，WK，WV矩阵
        # 并将结果拆成h块，然后将第二个和第三个维度值互换
        # q, k, v 的 shape 为： [batch_size, n_heads, len_q, d_k]
        query = self.L_Q(query).view(n_b, -1, self.h, self.d_k).transpose(1, 2)
        key = self.L_K(key).view(n_b, -1, self.h, self.d_k).transpose(1, 2)
        value = self.L_V(value).view(n_b, -1, self.h, self.d_k).transpose(1, 2)
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = Attention()(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来
        x = x.transpose(1, 2).contiguous().view(n_b, -1, self.h * self.d_k)  # x: [batch_size, n_heads,len_q*d_k]
        # 进行层归一化 和 残差连接
        return nn.LayerNorm(self.d_model)(self.fc(x).to(DEVICE) + residual.to(DEVICE))

# 前馈网络
class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x):  # x: [batch_size, seq_len, d_model]
        output = self.fc(x) # 进行两个的线性层
        # 进行层归一化 和 残差连接
        return nn.LayerNorm(self.d_model)(output.to(DEVICE) + x.to(DEVICE))  # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    # 编码层 包括一个多头注意力机制 + 一个前馈神经网络，每一个后都要执行残差连接和层归一化
    def __init__(self, n_heads, n_hidden, d_model=512):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, n_heads)
        self.feed_forward = FeedForwardNet(d_model, n_hidden)
        self.d_model = d_model

    def forward(self, x, mask):
        # 这里的x是经过嵌入层和位置编码后的数据
        x = self.self_attn(x, x, x, mask)
        # 注意到attn得到的结果x直接作为了下一层的输入
        x = self.feed_forward(x)
        return x

class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 嵌入层
        self.src_emb = nn.Embedding(args.src_vocab_size, args.d_model)
        # 位置编码
        self.pos_emb = PositionalEncoding(d_model=args.d_model, dropout=0.1)
        # 复制N个encoder layer, 一个解码器要包含N个解码层。每个解码层的输入和输出保持相同的尺寸。
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def forward(self, x, mask):
        x = self.src_emb(x)  # [batch_size, src_len, d_model]
        x = self.pos_emb(x)  # [batch_size, src_len, d_model]
        # 以此执行六层编码
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(DecoderLayer, self).__init__()
        # Self-Attention
        self.self_attn = MultiHeadedAttention(d_model, n_heads)
        # 与Encoder传入的 Context 进行Attention
        self.src_attn = MultiHeadedAttention(d_model, n_heads)
        self.feed_forward = FeedForwardNet(d_model, d_ff)

    def forward(self, x, context, src_mask, tgt_mask):
        # context来存放encoder 输出的结果
        # 注意self-attention的q，k和v均为decoder的输入经过嵌入和位置编码的结果
        x = self.self_attn(x, x, x, tgt_mask)
        # 注意context-attention的q为解码器上一个MultiHeadedAttention的输出，而k和v为编码器的输出
        x = self.src_attn(x, context, context, src_mask)
        x = self.feed_forward(x)
        return x

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(args.src_vocab_size, args.d_model)
        self.pos_emb = PositionalEncoding(d_model=args.d_model, dropout=0.1)
        # 复制N个encoder layer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.tgt_emb(x)  # [batch_size, tgt_len, d_model]
        x = self.pos_emb(x)  # [batch_size, tgt_len, d_model]
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(EncoderLayer(h, d_ff, d_model).to(DEVICE), N).to(DEVICE)
        self.decoder = Decoder(DecoderLayer(d_model, h, d_ff).to(DEVICE), N).to(DEVICE)
        self.generator = Generator(d_model, tgt_vocab)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

# 对Transformer的输出结果进行一次全连接，将输出映射到词表大小，然后通过SoftMax找出概率最大的那个座位越策结果
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)

def init_model_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # 这里初始化采用的是nn.init.xavier_uniform
    return model.to(DEVICE)

# 批量贪婪搜索
def batch_greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    batch_size, src_seq_len = src.size()  # src: [batch_size, sent_len]
    results = [[] for _ in range(batch_size)]
    stop_flag = [False for _ in range(batch_size)]
    count = 0
    memory = model.encode(src, src_mask) # 对源数据进行编码
    tgt = torch.Tensor(batch_size, 1).fill_(start_symbol).type_as(src.data)
    for s in range(max_len):
        www = (tgt != 0).unsqueeze(-2)
        tgt_mask = get_future_mask(tgt.size(1)).expand(batch_size, -1, -1).type_as(www.data) # 上三角形的mask，盖住后面的词
        out = model.decode(memory, src_mask, Variable(tgt), Variable(tgt_mask)) # 进行解码
        prob = model.generator(out[:, -1, :])
        pred = torch.argmax(prob, dim=-1) #  找出最大概率的值
        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1) # 把预测的值合并到已有的预测中去，作为下一次解码器的输入
        pred = pred.cpu().numpy()
        for i in range(batch_size):
            if stop_flag[i] is False:
                if pred[i] == end_symbol:
                    count += 1
                    stop_flag[i] = True
                else:
                    results[i].append(pred[i].item())
            if count == batch_size:
                break
    return results


