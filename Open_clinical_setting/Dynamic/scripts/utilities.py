import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random

# %matplotlib inline

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

    return filenames, labels


def read_csv_2(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    rid_list = [int(float(a[-2])) for a in your_list[1:]]
    viscode_list = [str(a[-1]) for a in your_list[1:]]
    filenames = [a[0] for a in your_list[1:]]
    # labels = [0 if a[1]=='NL' else 1 for a in your_list[1:]]
    labels = [int(float(a[1])) for a in your_list[1:]]

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
    with open('lookcsv/train_npy.csv', 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    labels, train_valid = your_list[0:1], your_list[1:79]

    random.shuffle(train_valid)
    with open('lookcsv/train.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(labels + train_valid[:55])
    with open('lookcsv/valid.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(labels + train_valid[55:])


def write_raw_score(f, preds, labels):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')




if __name__ == '__main__':

    f = open('raw_score.txt', 'w')
    preds = []
    labels = []
    write_raw_score(f, preds, labels)
    # get_imagename('data/data_110_4/', 'lookcsv/fcn_fcn1.csv')

    # read_csv('lookcsv/train_npy.csv')

    # data_split()