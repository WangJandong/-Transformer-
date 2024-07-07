import torch
import sacrebleu

def bleu_score(hyp, refs):
    bleu = sacrebleu.corpus_bleu(hyp, refs, tokenize='zh')
    return round(bleu.score, 2)

def bleu_score(hyp, refs):
    bleu = sacrebleu.corpus_bleu(hyp, refs, tokenize='zh')
    return round(bleu.score, 2)

def get_padding_mask(x, padding_idx):
    # 比较输入张量每个元素与填充索引，得到填充位置的掩码
    mask = (x == padding_idx)
    # 在最前面新增一个维度，用于后续与注意力得分矩阵广播相乘
    return mask.unsqueeze(1).byte()


# 参数 size 为句子长度
def get_subsequent_mask(size):
    # 1为batch维度
    mask_shape = (1, size, size)
    return 1-torch.tril(torch.ones(mask_shape)).byte()



