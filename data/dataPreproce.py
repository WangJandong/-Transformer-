import json
from collections import Counter
import jieba
import re
from translation.config.Config import *


# 中文分词
def divided_zh(sentence):
    return jieba.lcut(sentence)

# 英文分词
def divided_en(sentence):
    # 使用正则表达式匹配单词和标点符号
    pattern = r'\w+|[^\w\s]'
    return re.findall(pattern, sentence)

# 获取词表
def get_vocab(lang='en'):
    if lang == 'en':
        file_path = EN_VOCAB_PATH
    elif lang == 'zh':
        file_path = ZH_VOCAB_PATH

    with open(file_path, encoding='utf-8') as file:
        lines = file.read()

    id2vocab = lines.split('\n')
    vocab2id = {v: k for k, v in enumerate(id2vocab)}
    return id2vocab, vocab2id

# 生成词表
def generate_vocab():
    en_vocab = ['<pad>', '<unk>', '<sos>', '<eos>']
    zh_vocab = ['<pad>', '<unk>', '<sos>', '<eos>']
    zh_vocab_list = []
    en_vocab_list = []

    # 解析json文件
    with open(TRAIN_SAMPLE_PATH, encoding='utf-8') as file:
        lines = json.loads(file.read())
        for en_sent, zh_sent in lines:
            en_vocab_list += divided_en(en_sent)
            zh_vocab_list += divided_zh(zh_sent)
    print('train_sample count:', len(lines))

    # 按次数生成词表，此处可以去掉生僻词
    zh_vocab_kv = Counter(zh_vocab_list).most_common()
    zh_vocab += [k.lower() for k, v in zh_vocab_kv]

    en_vocab_kv = Counter(en_vocab_list).most_common()
    en_vocab += [k.lower() for k, v in en_vocab_kv]

    print('en_vocab count:', len(en_vocab))
    print('zh_vocab count:', len(zh_vocab))

    # 生成词表文件
    with open(ZH_VOCAB_PATH, 'w', encoding='utf-8') as file:
        file.write('\n'.join(zh_vocab))

    with open(EN_VOCAB_PATH, 'w', encoding='utf-8') as file:
        file.write('\n'.join(en_vocab))
