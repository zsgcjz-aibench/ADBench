import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import interp
import math

def write_csv(file_name, data):
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
            

def read_csv(data_file_path):
    data = []
    with open(data_file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        # print("data:", data)
        data = np.asarray(data)
        # print("data####:", data)
    return data

def read_csv_1(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames = [a[0] for a in your_list[1:]]
    # labels = [0 if a[1]=='NL' else 1 for a in your_list[1:]]
    labels = [int(float(a[1])) for a in your_list[1:]]
    rid_list = [int(float(a[-2])) for a in your_list[1:]]
    viscode_list = [a[-1] for a in your_list[1:]]

    return filenames, labels, rid_list, viscode_list

#给师兄的数据需要
def read_csv_2(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    rid_list = [int(float(a[0])) for a in your_list[1:]]
    viscode_list = [str(a[1]) for a in your_list[1:]]
    filenames = [a[2] for a in your_list[1:]]
    # labels = [0 if a[1]=='NL' else 1 for a in your_list[1:]]
    labels = [int(float(a[3])) for a in your_list[1:]]

    return rid_list, viscode_list, filenames, labels

def imshow(images, labels, full_path):              
    plt.figure(figsize=(5,5))
        
    temp = images[0,:,:,:].numpy()
    temp = np.transpose(temp, [1,2,0])
    plt.imshow(temp)
    plt.title(labels[0])
    print('file_name:', full_path[0])


def assemble_labels(step, y_true, y_pred, label, out):
    if(step==0):
        y_true = label
        y_pred = out
    else:
#         y_true = np.concatenate((y_true, label), 0)
#         y_pred = np.concatenate((y_pred, out), 0)
        y_true = torch.cat((y_true, label), 0)
        y_pred = torch.cat((y_pred, out))
    return y_true, y_pred

def get_imagename(filename, csvpath):
    pathlist = os.listdir(filename)
    print("pathlist:", pathlist)

    adni = pd.read_csv(csvpath)

    dataP = pd.DataFrame()
    for i in pathlist:
        name = i.split('.npy')[0]
        for index, row in adni.iterrows():
            imagename = row['filename']
            if name == imagename:
                dataP = dataP.append(row)

    dataP.to_csv('lookcsv/train_npy.csv', index=False)


def data_split():
    with open('../mci_test_data_adni_image.csv', 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    labels, train_valid = your_list[0:1], your_list[1:900]

    random.shuffle(train_valid)
    with open('lookcsv/mci_test.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(labels + train_valid[:475])
    with open('lookcsv/mci_t.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(labels + train_valid[475:])


def write_raw_score(f, preds, labels):
    # preds = preds.data.cpu().numpy()
    # labels = labels.data.cpu().numpy()
    preds = preds
    labels = labels
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')


def softmax(a, b):
    Max = max([a, b])
    a, b = a-Max, b-Max
    return math.exp(b) / (math.exp(a) + math.exp(b))


def read_raw_score(txt_file):
    labels, scores = [], []
    preds = []
    matrix = [[0, 0], [0, 0]]
    sens, spcs = [], []
    with open(txt_file, 'r') as f:
        for line in f:
            # print("line:", line)
            nl, ad, label = map(float, line.strip('\n').split('__'))
            pred = [nl, ad]
            if np.amax(pred) == pred[0]:
                if label == 0:
                    matrix[0][0] += 1
                if label == 1:
                    matrix[0][1] += 1
            elif np.amax(pred) == pred[1]:
                if label == 0:
                    matrix[1][0] += 1
                if label == 1:
                    matrix[1][1] += 1
            scores.append(softmax(nl, ad))  #  softmax：取概率值
            labels.append(int(label))
    print("labels:", len(labels))
    # ma = get_confusion_matrix(preds, labels)
    print("matrix:", matrix)
    accu = float(matrix[0][0] + matrix[1][1])/ float(sum(matrix[0]) + sum(matrix[1]))

    TN, FP, FN, TP = matrix[1][1], matrix[0][1], matrix[1][0], matrix[0][0]
    sens.append(TP / (TP + FN))  # 当前数据集的sens
    spcs.append(TN / (TN + FP))

    print("accu#####:", np.mean(accu))  # 当前数据集的准确率
    print("sens#####:", np.mean(sens))
    print("spcs#####:", np.mean(spcs))
    return np.array(labels), np.array(scores), accu, sens, spcs

def get_roc_info(y, y_score_list):
    fpr_pt = np.linspace(0, 1, 1001)
    tprs, aucs = [], []
    for y_score in y_score_list:
        fpr, tpr, _ = roc_curve(y_true=y, y_score=y_score, drop_intermediate=True)
        tprs.append(interp(fpr_pt, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
    tprs_mean = np.mean(tprs, axis=0)
    tprs_std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(tprs_mean + tprs_std, 1)
    tprs_lower = np.maximum(tprs_mean - tprs_std, 0)
    auc_mean = auc(fpr_pt, tprs_mean)
    print("auc_mean:", auc_mean)
    auc_std = np.std(aucs)
    auc_std = 1 - auc_mean if auc_mean + auc_std > 1 else auc_std

    rslt = {'xs': fpr_pt,
            'ys_mean': tprs_mean,
            'ys_upper': tprs_upper,
            'ys_lower': tprs_lower,
            'auc_mean': auc_mean,
            'auc_std': auc_std}

    return rslt


if __name__ == '__main__':


    # test_y_true = [1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1., 1.]
    # test_y_pred = [[ 0.1367,  0.3507],
    #     [ 0.6175, -0.7439],
    #     [ 0.0058,  0.2756],
    #     [-0.1082,  0.2260],
    #     [ 0.0397,  0.0774],
    #     [ 0.0282,  0.1790],
    #     [ 0.0779,  0.2546],
    #     [-0.0901,  0.2256],
    #     [ 0.0389,  0.1652],
    #     [-0.0285,  0.2460],
    #     [ 0.0810,  0.1235],
    #     [ 0.2419,  0.0420],
    #     [ 0.0230,  0.1645],
    #     [ 0.0396,  0.1633],
    #     [ 0.1149,  0.0349],
    #     [-0.0275,  0.0820],
    #     [ 0.1173,  0.0375]]
    #
    # test_y_true = torch.tensor(test_y_true, dtype=torch.float64)
    # test_y_pred = torch.tensor(test_y_pred)

    # print("test_y_true:", test_y_true)
    #
    # print("test_y_pred:", test_y_pred)
    # f = open('raw_score.txt', 'w')
    # write_raw_score(f, test_y_pred, test_y_true)

    labels, scores, accu, sens, spcs = read_raw_score('raw_score_ResNet.txt')
    print(labels)
    print(scores)
    #
    # labelsL = []
    scoresL = []
    # # labelsL.append(labels)
    scoresL.append(scores)
    get_roc_info(labels, scoresL)

    labelsd, scoresd, accud, sensd, spcsd = read_raw_score('raw_score_dynamic.txt')
    #
    # print(labelsd)
    # print(len(scoresd))




    # get_imagename('data/', 'lookcsv/fcn_fcn1.csv')

    # read_csv('lookcsv/train_npy.csv')

    # data_split()