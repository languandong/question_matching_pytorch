import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd

def predict(config, test_iter):
    model = torch.load(config.read_model_file)
    print("read model from {}".format(config.read_model_file))
    model.to(config.device)
    model.eval()

    preds = []
    for input_ids, attention_mask, token_type_ids, labels, tokens_cpu in tqdm(test_iter, position=0, ncols=80,
                                                                              desc='测试中'):
        prob, pred = model.forward(input_ids, attention_mask, token_type_ids)

        # labels = labels.cpu().numpy()
        # true_doc_label_list.extend(labels)

        pred = pred.cpu().numpy()
        preds.extend(pred)

    # f1 = f1_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    # p = precision_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    # r = recall_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    # acc = accuracy_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list)
    # print('acc: {}, precision: {}, recall {}, f1 {}'.format(acc, p, r, f1))
    pd.DataFrame({'label': [i for i in preds]}).to_csv(
        config.source_data_dir + "/result.csv", index=False, header=None)

