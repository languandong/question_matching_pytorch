import sys
sys.path.append('..')
from config import Config
import pandas as pd

# 定义单个数据样本对象
class Example(object):
    def __init__(self, textA, textB, label):
        self.textA = textA
        self.textB = textB
        self.label = label

# 创建数据
def create_data(file_path):
    def load_data(file_path):
        """读取数据"""
        doc_type = file_path.split('.')[-1]
        df = pd.read_csv(file_path) if doc_type == 'csv' else pd.read_csv(file_path,sep='\t',header=None)
        lines = []
        for _, item in df.iterrows():
            # df.iterrows返回每行索引和每行内容
            if len(item) == 3:
                lines.append((item['text_A'], item['text_B'], item['label']))
            else:
                lines.append((item[0], item[1], -100))

        return lines

    # 存储数据对象
    data = []
    # 读入整份数据
    lines = load_data(file_path)
    # 把单个数据对象依次存入
    for index, line in enumerate(lines):
        textA = line[0]
        textB = line[1]
        label = line[2]
        data.append(Example(textA=textA, textB=textB, label=label))

    return data


if __name__ == '__main__':
    config = Config()
    data = create_data(config.source_data_dir + 'test_A.tsv')
    raw_data = pd.read_csv(config.source_data_dir + 'test_A.tsv',sep='\t',header=None)
    print(len(data))
    print(len(raw_data))