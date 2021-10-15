import torch
from torch import nn
from transformers import BertModel

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pre_model = BertModel.from_pretrained(config.pretrain_model_path)
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.use_pooling = config.use_pooling
        self.label_num = config.label_num
        self.fc = torch.nn.Linear(config.bert_hidden, self.label_num)
        self.cross_entropy = nn.functional.cross_entropy

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        sequence_out, ori_pooled_output, encoded_layers = self.pre_model(input_ids, attention_mask, token_type_ids)
        sequence_out = self.dropout(sequence_out)

        if self.use_pooling == "max":
            pass
        elif self.use_pooling == "avg":
            pass
        else:
            # 只取[cls]的向量
            sequence_out = sequence_out[:, 0, :]

        # 最后的全连接层
        logits = self.fc(sequence_out)

        # 返回计算的损失函数值
        # 训练
        if labels is not None:
            loss = self.cross_entropy(logits, labels)
            return loss
        # 预测
        else:
            prob = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            return prob, pred
