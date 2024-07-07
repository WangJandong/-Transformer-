from torch.nn.utils.rnn import pad_sequence
from translation.data.dataPreproce import *
from translation.net_transformer.transformer_model import  make_model
from translation.net_transformer.Utils import get_padding_mask,get_subsequent_mask,bleu_score



def batch_greedy_decode(model, src_x, src_mask, max_len=50):

    model_mod = model.module if MULTI_GPU else model

    zh_id2vocab, _ = get_vocab('zh')
    memory = model_mod.encoder(src_x, src_mask)
    # 初始化目标值
    prob_x = torch.tensor([[SOS_ID]] * src_x.size(0))
    prob_x = prob_x.to(DEVICE)

    for _ in range(max_len):
        prob_mask = get_padding_mask(prob_x, PAD_ID)
        output = model_mod.decoder(prob_x, prob_mask, memory, src_mask)
        output = model_mod.generator(output[:, -1, :])
        predict = torch.argmax(output, dim=-1, keepdim=True)
        prob_x = torch.concat([prob_x, predict], dim=-1)
        # 全部预测结束，结束循环
        if torch.all(predict==EOS_ID).item():
            break
    # 根据预测值id，解析翻译后的句子
    batch_prob_text = []
    for prob in prob_x:
        prob_text = []
        for prob_id in prob:
            if prob_id == SOS_ID:
                continue
            if prob_id == EOS_ID:
                break
            prob_text.append(zh_id2vocab[prob_id])
        batch_prob_text.append(''.join(prob_text))
    return batch_prob_text

if __name__ == '__main__':

    en_id2vocab, en_vocab2id  = get_vocab('en')
    zh_id2vocab, zh_vocab2id  = get_vocab('zh')

    SRC_VOCAB_SIZE = len(en_id2vocab)
    TGT_VOCAB_SIZE = len(zh_id2vocab)

    model = make_model(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, N_HEAD, D_FF, N, DROPOUT)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR + '/best_model.pth', map_location=DEVICE))

    model.eval()

    texts = [
        "we got the book.",
    ]

    batch_src_token = [[en_vocab2id.get(v.lower(), UNK_ID) for v in divided_en(text)] for text in texts]
    batch_src = [torch.LongTensor([SOS_ID]+src+[EOS_ID]) for src in batch_src_token]
    src_x = pad_sequence(batch_src, True, PAD_ID)
    src_mask = get_padding_mask(src_x, PAD_ID)

    src_x = src_x.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    prob_sent = batch_greedy_decode(model, src_x, src_mask)
    print("中文：",texts)
    print("英文：",prob_sent)