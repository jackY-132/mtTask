import numpy as np
import torch
from utils.utils import get_hyperparameter, get_device, get_en_tokenizer, get_ch_tokenizer
from model import Transformer
from beam_decoder import beam_search

args = get_hyperparameter()
DEVICE = get_device(args)

# 单句预测
def translate(src, model):
    chin_tok = get_ch_tokenizer()
    with torch.no_grad():
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        pred, _ = beam_search(model, src, src_mask, args.max_len,0, 2, 3,args.beam_size, DEVICE)
        pred = [h[0] for h in pred]
        translation = [chin_tok.decode_ids(_s) for _s in pred]
        print(translation[0])


sent = "It gets up to about 150 feet long."
# sent = " Now in our town, where the volunteers supplement a highly skilled career staff, you have to get to the fire scene pretty early to get in on any action."
# tgt_sent = " 在我们的小镇 在一个志愿者都是成功人士的地方 你必须要很早到现场 才有可能加入战况 "
model_dict = torch.load("/tmp/nlp/work/model.pth", map_location=torch.device("cpu"))
model = Transformer(args.tgt_vocab_size, args.n_layers, args.d_model, args.d_ff, args.n_heads, args.dropout).to(DEVICE)
model.load_state_dict(model_dict)
src_tokens = [[2] + get_en_tokenizer().EncodeAsIds(sent) + [3]]
batch_input = torch.LongTensor(np.array(src_tokens)).to(args.device)
translate(batch_input, model)
