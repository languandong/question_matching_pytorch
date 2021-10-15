from transformers import BertTokenizer
import torch
import numpy as np
import os
import datetime
from total_utils.common import get_logger

class Config(object):
    def __init__(self):
        self.gpu_id = 1
        self.use_multi_gpu = False
        self.use_pooling = ["max", "avg", "None"][2]

        # train device selection
        if self.use_multi_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
                print('current device:', torch.cuda.current_device())  # watch for current device
                n_gpu = 1
                self.n_gpu = n_gpu

        # 基本参数
        self.bert_hidden = 768
        self.train_epoch = 10
        self.random_seed = 2021
        self.batch_size = 32
        self.sequence_length = 128
        self.dropout_rate = 0.1
        self.label_num = 2
        # 学习率衰减参数
        self.warmup_prop = 0.1
        self.clip_grad = 2.0
        self.bert_learning_rate = 5e-5

        self.origin_data_dir = "E:/deepLearning/sentences_pair/datasets/"
        self.source_data_dir = "E:/deepLearning/sentences_pair/datasets/source_data/"
        self.model_save_path = "E:/deepLearning/sentences_pair/model_save/"
        self.config_file_path = "E:/deepLearning/sentences_pair/config.py"


        self.pretrain_model_path = "E:/deepLearning/sentences_pair/pytorch_bert_chinese_L-12_H-768_A-12/"
        self.tokenizer = BertTokenizer(vocab_file=self.pretrain_model_path + "vocab.txt", do_lower_case=True)

    def train_init(self):
        # 保证实验结果的可复现性
        # 设定唯一的种子，确保每次训练数据的输入相同
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        self.get_save_path()

    def get_save_path(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.model_save_path = self.model_save_path + "bert_use_pooling_{}_{}".format(self.use_pooling, timestamp)

        # 创建文件夹
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # 将config.py文件写入
        with open(self.model_save_path + "/config.test", "w", encoding="utf8") as fw:
            with open(self.config_file_path, "r", encoding="utf8") as fr:
                content = fr.read()
                fw.write(content)
        # 写入Logger
        self.logger = get_logger(self.model_save_path + "/log.log")
        self.logger.info('current device:{}'.format(torch.cuda.current_device()))  # watch for current device
