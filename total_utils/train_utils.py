import os
import sys
sys.path.append('..')
from model_base import Model
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score


def train(config, train_iter, dev_iter):
    model = Model(config).to(config.device)
    bert_param_optimizer = list(model.pre_model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params':[param for n,param in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': config.bert_learning_rate
        },
        {
            'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            "lr": config.bert_learning_rate
        }
    ]
    optimizer = AdamW(params = optimizer_grouped_parameters,
                      betas = (0.9, 0.98),
                      lr = config.bert_learning_rate,
                      eps = 1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                    num_warmup_steps = int(len(train_iter) * config.train_epoch * config.warmup_prop),
                                    num_training_steps = len(train_iter) * config.train_epoch)

    # 训练
    cum_step = 0
    for i in range(config.train_epoch):
        model.train()
        for input_ids, attention_mask, token_type_ids, labels, tokens_cpu in tqdm(train_iter, position=0, ncols=80, desc='训练中'):
            loss = model.forward(input_ids, attention_mask, token_type_ids, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            cum_step +=  1

        # 验证
        f1, p, r = set_eval(config, model, dev_iter)

        # 保存模型
        model_to_save = model.module if hasattr(model,'module') else model
        output_model_file = os.path.join(
            os.path.join(config.model_save_path, 'model_{:.4f}_{:.4f}_{:.4f}_{}.bin'.format(p, r, f1, str(cum_step)))
        )
        torch.save(model_to_save, output_model_file)

def set_eval(config, model, dev_iter):
    model.eval()
    true_label_list, pred_label_list = [], []
    for input_ids, attention_mask, token_type_ids in tqdm(dev_iter, position=0, ncols=80, desc='验证中'):
        prob, pred = model.forward(input_ids, attention_mask, token_type_ids, labels=None)

        labels = labels.cpu().numpy()
        pred = pred.cpu().numpy()
        true_label_list.extend(labels)
        pred_label_list.extend(pred)

    # 评价指标
    f1 = f1_score(y_true=true_label_list, y_pred=pred_label_list, average='macro')
    p = precision_score(y_true=true_label_list, y_pred=pred_label_list, average='macro')
    r = recall_score(y_true=true_label_list, y_pred=pred_label_list, average='macro')

    # 评估结果写入logger
    config.logger.info(report)
    config.logger.info('precision: {}, recall {}, f1 {}'.format(p, r, f1))

    return f1, p, r