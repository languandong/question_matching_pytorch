from dataLoad import create_data
import numpy as np
import sys
sys.path.append('..')
from config import Config
import torch

class DataIterator(object):
    def __init__(self, config, data_file, is_test=False):
        self.batch_size = config.batch_size
        self.seq_length = config.sequence_length
        self.tokenizer = config.tokenizer
        self.device = config.device
        # 数据的操作
        self.data = create_data(data_file)  # 样本数据对象数组
        self.num_records = len(self.data)  # 数据的个数
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test

        self.data_type = data_file.split('/')[-1].split('.')[0]
        # 训练模式下数据乱序
        if not self.is_test:
            np.random.shuffle(self.all_idx)

        print(self.data_type,"样本个数：", self.num_records)

    def convert_single_example(self, example_idx):
        '''处理单个数据样本'''
        textA_list = list(self.data[example_idx].textA)
        textB_list = list(self.data[example_idx].textB)
        label = self.data[example_idx].label
        # 将文本转为词典索引
        tokens_A = []
        tokens_B = []
        for index, token in enumerate(textA_list):
            char_list = self.tokenizer.tokenize(token.lower())
            if char_list:
                tokens_A.append(char_list[0])
            else:
                # 解析不到的特殊字符用 §替代
                tokens_A.append("§")
        for index, token in enumerate(textB_list):
            char_list = self.tokenizer.tokenize(token.lower())
            if char_list:
                tokens_B.append(char_list[0])
            else:
                # 解析不到的特殊字符用 §替代
                tokens_B.append("§")
        # 加入修饰符
        tokens = ["[CLS]"] + tokens_A + ["[SEP]"] + tokens_B + ["[SEP]"]
        # 将序列文字转为字典索引
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 初始化单个样本的各个状态向量
        input_ids_cpu = np.zeros((1, self.seq_length), dtype=np.int64)
        attention_mask_cpu = np.zeros((1, self.seq_length), dtype=np.int64)
        token_type_ids_cpu = np.zeros((1,self.seq_length), dtype=np.int64)
        label_cpu = np.zeros((1,), dtype=np.int64)
        tokens_cpu = np.array(['*NULL*'] * self.seq_length)
        tokens_cpu = np.expand_dims(tokens_cpu, axis=0)

        # 复制单个样本的状态向量
        # 个别样本长度超出作截断
        input_ids_cpu[0, :len(input_ids)] = input_ids if len(input_ids) <= self.seq_length else input_ids[:self.seq_length]
        attention_mask_cpu[0, :len(input_ids)] = [1] * len(input_ids) if len(input_ids) <= self.seq_length else [1] * self.seq_length

        if len(input_ids) <= self.seq_length:
            token_type_ids_cpu[0, (len(tokens_A) + 2): len(input_ids)] = 1
            tokens_cpu[0, :len(tokens)] = tokens
        else:
            token_type_ids_cpu[0, (len(tokens_A) + 2):] = 1
            tokens_cpu[0,:] = tokens[:self.seq_length]

        label_cpu[0] = label

        # input_ids_cpu[0, :len(input_ids)] = input_ids
        # attention_mask_cpu[0, :len(input_ids)] = [1] * len(input_ids)
        # token_type_ids_cpu[0, (len(tokens_A) + 2): len(input_ids)] = 1
        # label_cpu[0] = label
        # tokens_cpu[0, :len(tokens)] = tokens

        return input_ids_cpu, attention_mask_cpu, token_type_ids_cpu, label_cpu, tokens_cpu

    def __iter__(self):
        return self

    def __next__(self):
        # 初始化存储一个batch状态向量的变量
        input_ids_cpu, attention_mask_cpu, token_type_ids_cpu, label_cpu, tokens_cpu = [], [], [], [], []
        example_count = 0

        # 迭代数量超出总数量时
        if self.idx >= self.num_records:
            self.idx = 0
            if not self.is_test:
                np.random.shuffle(self.all_idx)
            raise StopIteration

        # 返回一个batch的数据
        while example_count < self.batch_size:
            # 依次读取乱序后的数据样本
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)

            single_input_ids, single_attention_mask, single_token_type_ids, single_label, single_tokens = res

            # 把样本逐个堆叠成一个batch的数据
            if example_count == 0:
                input_ids_cpu, attention_mask_cpu, token_type_ids_cpu, label_cpu, tokens_cpu = \
                    single_input_ids, single_attention_mask, single_token_type_ids, single_label, single_tokens
            else:
                input_ids_cpu = np.concatenate((input_ids_cpu, single_input_ids), axis=0)
                attention_mask_cpu = np.concatenate((attention_mask_cpu, single_attention_mask), axis=0)
                token_type_ids_cpu = np.concatenate((token_type_ids_cpu, single_token_type_ids), axis=0)
                label_cpu = np.concatenate((label_cpu, single_label), axis=0)
                tokens_cpu = np.concatenate((tokens_cpu, single_tokens), axis=0)

            example_count += 1
            self.idx += 1
            if self.idx >= self.num_records:
                break

        input_ids = torch.from_numpy(input_ids_cpu).to(self.device)
        attention_mask = torch.from_numpy(attention_mask_cpu).to(self.device)
        token_type_ids = torch.from_numpy(token_type_ids_cpu).to(self.device)
        labels = torch.from_numpy(label_cpu).to(self.device)

        return input_ids, attention_mask, token_type_ids, labels, tokens_cpu

    def __len__(self):
        if self.num_records % self.batch_size == 0:
            return self.num_records // self.batch_size
        else:
            return self.num_records // self.batch_size + 1


if __name__ == '__main__':
    config = Config()
    test_iter = DataIterator(config, config.source_data_dir + 'test.csv', is_test=True)
    print(len(test_iter))
    for input_ids, attention_mask, token_type_ids, labels, tokens_list in test_iter:
        print(input_ids)
        print(attention_mask)
        print(token_type_ids)
        print(labels)
        print(tokens_list)





