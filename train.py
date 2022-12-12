import copy
import torch
import sacrebleu
from tqdm import tqdm
from beam_decoder import beam_search
from model import batch_greedy_decode
from utils.utils import get_ch_tokenizer, get_hyperparameter
args = get_hyperparameter()


def train(train_data, dev_data, model, criterion, optimizer):
    best_bleu_score = 0
    save_dir = "/tmp/nlp/save_models/save_self/"
    for epoch in range(args.epoch_num):
        loss_sum = 0.
        step = 0
        for step, batch in tqdm(enumerate(train_data, start=1)):
            model.train()  # 设置train mode
            optimizer.zero_grad()  # 梯度清零
            # forward
            outputs = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            outputs = model.generator(outputs)
            loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), batch.trg_y.contiguous().view(-1))  # loss: [batch_size * tgt_len]
            loss_sum += loss
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
        print("Epoch: {}, loss: {}".format(epoch, loss_sum/step))
        val_loss_sum = 0.
        val_step = 0
        for val_step, batch_dev in tqdm(enumerate(dev_data, start=1)):
            val_loss = validate_step(model, batch_dev.src, batch_dev.trg, batch_dev.trg_y, batch_dev.src_mask, batch_dev.trg_mask, criterion)
            val_loss_sum += val_loss
            # val_metric_sum += acc
        bleu_score = evaluate(dev_data, model)
        print('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, val_loss_sum/val_step, bleu_score))
        # 保存模型
        if bleu_score > best_bleu_score:
            print("-------- Save Best Model! --------")
            best_bleu_score = bleu_score
            checkpoint = save_dir + 'best_model.pth'
            model_sd = copy.deepcopy(model.state_dict())
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
            }, checkpoint)

def validate_step(model, inp, targ, targ_real, src_mask, trg_mask, criterion):
    model.eval()
    with torch.no_grad():
        prediction = model(inp, targ, src_mask, trg_mask)
        prediction = model.generator(prediction)
        val_loss = criterion(prediction.contiguous().view(-1, prediction.size(-1)), targ_real.contiguous().view(-1))
    return val_loss.item()

def test(data, model, criterion):
    test_loss_sum = 0.
    test_step = 0
    for test_step, batch in tqdm(enumerate(data, start=1)):
        loss = validate_step(model, batch.src, batch.trg, batch.trg_y, batch.src_mask, batch.trg_mask, criterion)
        test_loss_sum += loss
    print("loss: {}".format(test_loss_sum/test_step))

def evaluate(data, model, mode='dev', use_beam=False):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_chn = get_ch_tokenizer()
    trg = []
    res = []
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):
            # 对应的中文句子
            cn_sent = batch.trg_text # 所有的中文句子
            src = batch.src   # [batch_size, sent_len]
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask, args.max_len, args.padding_idx, args.bos_idx, args.eos_idx, args.beam_size, args.device)
            else:
                decode_result = batch_greedy_decode(model, src, src_mask, max_len=args.max_len)
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]  # 将编号转为文字
            trg.extend(cn_sent)
            res.extend(translation)
    if mode == 'test':
        with open(args.output_path, "w") as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + trg[i] + '|||' + res[i] + '\n'
                fp.write(line)
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)



