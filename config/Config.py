import torch
import os

# 路径参数
BASE_PATH = os.path.dirname(__file__) # translation\config
TRAIN_SAMPLE_PATH = os.path.join(BASE_PATH, '..\\data\\inputs\\demo\\train.json') # 训练集合
DEV_SAMPLE_PATH = os.path.join(BASE_PATH, '..\\data\\inputs\\demo\\dev.json')    # 测试集合
ZH_VOCAB_PATH = os.path.join(BASE_PATH, '..\\data\\vocab\\zh.txt')         # 中文词表
EN_VOCAB_PATH = os.path.join(BASE_PATH, '..\\data\\vocab\\en.txt')         # 英文词表
MODEL_DIR = os.path.join(BASE_PATH, '..\\model\\checkpoints')              # 模型路径

# 定义特殊值
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

# 模型参数 学习率、训练轮数、批大小、GPU0内分配的批大小
LR = 1e-5
EPOCH = 100
BATCH_SIZE = 20
BATCH_SIZE_GPU0 = 5

# 超参数和网络规模相关
D_MODEL = 512
N_HEAD = 8
D_FF = 2048
N = 6
DROPOUT = 0.1

# 生成长度最大值
MAX_LEN = 50

LABEL_SMOOTHING = 0.1

# 设置 cuda or CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 是否是多核CPU
MULTI_GPU = False
if torch.cuda.device_count()>1:
    MULTI_GPU = True
