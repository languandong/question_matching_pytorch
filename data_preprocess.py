# from config import Config
import pandas as pd
import sys
sys.path.append('')
from config import Config
import numpy as np
import matplotlib; matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

config = Config()


def get_df(dataset, data_type):
    file_path = config.origin_data_dir + dataset + '/' + data_type
    text_A, text_B, label_list = [], [], []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            item = line.strip().split('\t')

            # 清理异常
            if item[0] == 'nan' or 'null' in item[0]:
                print(item)
                continue
            if item[1] == 'nan' or 'null' in item[1]:
                print(item)
                continue

            text_A.append(item[0])
            text_B.append(item[1])
            label_list.append(item[2])

    df = pd.DataFrame({
        'text_A': text_A,
        'text_B': text_B,
        'label': label_list
    })

    return df


def create_dataset(data_type):
    try:
        BQ_data = get_df('BQ', data_type)
        LCQMC_data = get_df('LCQMC', data_type)
        OPPO_data = get_df('OPPO', data_type)
    except FileNotFoundError:
        pass

    if data_type == 'test':
        data = BQ_data.append(LCQMC_data)
    else:
        data = BQ_data.append(LCQMC_data).append(OPPO_data)

    data.to_csv(config.source_data_dir + data_type + '.csv', index=False, encoding='utf-8')
    print(data_type + '数据生成完毕')


def count_length(file_path):
    data = pd.read_csv(file_path)
    data['length_1'] = data.text_A.apply(lambda x:len(x))
    data['length_2'] = data.text_B.apply(lambda x:len(x))
    data['length'] = data['length_1'] + data['length_2'] + 3
    print(data['length'].max())
    # print('begin')
    # plt.bar(list(range(0,len(data))), data['length'])
    # plt.show()
    # print('over')
    # return data.index

if __name__ == '__main__':
    # import os, sys
    # print(sys.path[0])
    create_dataset('train')
    create_dataset('dev')
    create_dataset('test')

    # count_length('./datasets/source_data/train.csv')
