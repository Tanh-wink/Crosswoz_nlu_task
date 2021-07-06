import os
import pickle
import json

def load_pkl_data(filePath):
    with open(filePath, 'rb') as fp:
        data_pkl = fp.read()
    print(f'loaded {filePath}')
    return pickle.loads(data_pkl)


def save_pkl_data(data, filePath):

    data_pkl = pickle.dumps(data)
    with open(filePath, 'wb') as fp:
        fp.write(data_pkl)
    print(f'saved {filePath}')

def check_dir(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def calculateF1(goldens, predicts):
    TP, FP, FN = 0, 0, 0
    for golden, preds in zip(goldens, predicts):
        for ele in preds:
            if ele in golden:
                TP += 1
            else:
                FP += 1
        for ele in golden:
            if ele not in preds:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, f1


def cal_multi_F1(goldens, predicts, label2id: dict):
    total_TP, total_FP, total_FN = 0, 0, 0
    total_f1 = 0
    label_info = {}
    for label in label2id.keys():
        TP, FP, FN = 0, 0, 0
        for golden, preds in zip(goldens, predicts):
            for ele in preds:
                if ele == label:
                    if ele in golden:
                        TP += 1
                    else:
                        FP += 1
            for ele in golden:
                if ele == label:
                    if ele not in preds:
                        FN += 1
        # 计算每一个类别的f1
        precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
        recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
        label_info[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        total_f1 += f1

        total_TP += TP
        total_FN += FN
        total_FP += FP
    # micro f1
    micro_precision = 1.0 * total_TP / (total_TP + total_FP) if total_TP + total_FP else 0.
    micro_recall = 1.0 * total_TP / (total_TP + total_FN) if total_TP + total_FN else 0.
    micro_f1 = 2.0 * micro_precision * micro_recall / (
                micro_precision + micro_recall) if micro_precision + micro_recall else 0.

    # macro f1
    macro_f1 = total_f1 / len(label2id)
    return micro_f1, macro_f1, label_info