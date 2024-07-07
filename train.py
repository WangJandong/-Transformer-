import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from translation.data.dataPreproce import *
from translation.net_transformer.transformer_model import  make_model
from translation.net_transformer.Utils import get_padding_mask,get_subsequent_mask,bleu_score
from translation.utils.data_parallel import BalancedDataParallel


def lr_lambda_fn(step, wramup):
    lr = 0
    if step <= wramup:
        lr = step / wramup * 10
    else:
        lr = wramup / step * 10
    return max(lr, 0.1)

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
def evaluate(loader, model, max_len=50):
    tgt_sent = []
    prob_sent = []

    for src_x, src_mask, tgt_x, tgt_mask, tgt_y, tgt_text in loader:
        src_x = src_x.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        batch_prob_text = batch_greedy_decode(model, src_x, src_mask, max_len)
        tgt_sent += tgt_text
        prob_sent += batch_prob_text

    print(prob_sent)
    print(tgt_sent)

    # 注意参考句子是多组
    return bleu_score(prob_sent, [tgt_sent])
def print_memory():
    # 获取当前可用的GPU数量
    num_gpus = torch.cuda.device_count()
    # 遍历每个GPU，输出GPU的占用情况
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_name(i)
        utilization = round(torch.cuda.max_memory_allocated(i) / 1024**3, 2)  # 显存使用量（以GB为单位）
        print(f"GPU {i}: {gpu}, Memory Utilization: {utilization} GB")

class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        if type == 'train':
            file_path = TRAIN_SAMPLE_PATH
        elif type == 'dev':
            file_path = DEV_SAMPLE_PATH
        # 读取文件
        with open(file_path, encoding='utf-8') as file:
            self.lines = json.loads(file.read())
        # 词表引入
        _, self.en_vocab2id = get_vocab('en')
        _, self.zh_vocab2id = get_vocab('zh')

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        en_text, zh_text = self.lines[index]
        source = [self.en_vocab2id.get(v.lower(), UNK_ID) for v in divided_en(en_text)]
        target = [self.zh_vocab2id.get(v.lower(), UNK_ID) for v in divided_zh(zh_text)]
        return source, target, zh_text
    def collate_fn(self, batch):
        batch_src, batch_tgt, tgt_text = zip(*batch)
        # source
        src_x = pad_sequence([torch.LongTensor([SOS_ID] + src + [EOS_ID]) for src in batch_src], True, PAD_ID)
        src_mask = get_padding_mask(src_x, PAD_ID)
        # target
        tgt_f = pad_sequence([torch.LongTensor([SOS_ID] + tgt + [EOS_ID]) for tgt in batch_tgt], True, PAD_ID)
        tgt_x = tgt_f[:, :-1]
        tgt_pad_mask = get_padding_mask(tgt_x, PAD_ID)
        tgt_subsqueent_mask = get_subsequent_mask(tgt_x.size(1))
        tgt_mask = tgt_pad_mask | tgt_subsqueent_mask
        tgt_y = tgt_f[:, 1:]
        # return
        return src_x, src_mask, tgt_x, tgt_mask, tgt_y, tgt_text

# 训练
def run_epoch(loader, model, loss_fn, optimizer=None):
    # 初始化loss值，和batch数量总数
    total_batchs = 0.
    total_loss = 0.
    # 加载数据并开始训练
    for src_x, src_mask, tgt_x, tgt_mask, tgt_y, tgt_text in loader:

        src_x = src_x.to(DEVICE)
        src_mask = src_mask.to(DEVICE)
        tgt_x = tgt_x.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)
        tgt_y = tgt_y.to(DEVICE)

        output = model(src_x, src_mask, tgt_x, tgt_mask)
        # 交叉熵损失，要求目标值是一维的
        loss = loss_fn(output.reshape(-1, output.shape[-1]), tgt_y.reshape(-1))
        # 累积batch数量和loss值
        total_batchs += 1
        total_loss += loss.item()

        # 如果有优化器，则表示需要反向传播
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 返回epoch的平均loss值
    return total_loss / total_batchs


if __name__ == '__main__':
    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   collate_fn=train_dataset.collate_fn)

    dev_dataset = Dataset('dev')
    dev_loader = data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dev_dataset.collate_fn)

    en_id2vocab, _  = get_vocab('en')
    zh_id2vocab, _ = get_vocab('zh')

    SRC_VOCAB_SIZE = len(en_id2vocab)
    TGT_VOCAB_SIZE = len(zh_id2vocab)

    model = make_model(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, N_HEAD, D_FF, N, DROPOUT)
    model = model.to(DEVICE)

    if MULTI_GPU:
        # model = nn.DataParallel(model)
        model = BalancedDataParallel(BATCH_SIZE_GPU0, model, dim=0)

    # print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    loss_fn = CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)
    optimizer = Adam(model.parameters(), lr=LR)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda_fn(step, EPOCH/4))

    best_bleu = 0

    for e in range(EPOCH):
        # 训练流程
        model.train()
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer)
        lr_scheduler.step()
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print("current_lr:",current_lr)

        # 验证流程
        model.eval()
        dev_loss = run_epoch(dev_loader, model, loss_fn, None)
        dev_bleu = evaluate(dev_loader, model, MAX_LEN)

        print('>> epoch:', e, 'train_loss:', round(train_loss, 6), 'dev_loss:', round(dev_loss, 6), 'dev_bleu:', dev_bleu)

        if dev_bleu > best_bleu:
            model_mod = model.module if MULTI_GPU else model
            torch.save(model_mod.state_dict(), MODEL_DIR+'/best_model.pth')
            best_bleu = dev_bleu

        # 调用
        print_memory()
        print('--' * 10)