import torch
from torch import nn
from torch.utils.data import Dataset
import csv
import numpy as np
import utilities as UT
import skimage.transform as skTrans
import random
import pandas as pd
import torchvision.models as models
import cv2
import nibabel as nib
import os
from NeurIPS.Data_silos_clinical_setting.random.utils.eval_index_cal import cal_CI_plot_close3, assemble_labels, cal_CI_plot_close

class LRP(nn.Module):
    """The model we use in the paper."""

    def __init__(self, dropout=0.4, dropout2=0.4):
        nn.Module.__init__(self)
        self.Conv_1 = nn.Conv3d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_1_mp = nn.MaxPool3d(2)
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_2_mp = nn.MaxPool3d(3)
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_3_mp = nn.MaxPool3d(2)
        self.Conv_4 = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.Conv_4_mp = nn.MaxPool3d(3)
        self.dense_1 = nn.Linear(2304, 64)
        self.dense_2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.dense_2(x)
        return x


def read_csv_complete1(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    # print("your_list:", your_list)
    filenames, labels, demors, rids, visits = [], [], [], [], []
    for line in your_list[1:]:

        filenames.append(line[8])

        if line[2] == 'CN':
            label = 0
        else:
            label = 1

        rid = int(float(line[6]))
        visit = line[9]


        rids.append(rid)
        visits.append(visit)
        labels.append(label)

        # print("###:", len(rids))
    return filenames, labels, rids, visits

class Dataset_Early_Fusion(Dataset):
    def __init__(self,filepath,
                 label_file):
        self.Data_list, self.Label_list, self.rid_list, self.visit_list = read_csv_complete1(label_file)
        self.filepaths = filepath

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):

        label = self.Label_list[idx]
        rid = self.rid_list[idx]
        visit = self.visit_list[idx]

        full_path = self.Data_list[idx]

        im = np.load(self.filepaths + full_path + '.npy')  # (110, 110, 110)
        im = np.reshape(im, (1, 182, 218, 182))

        return im, label, rid, visit  # output image shape [T,C,W,H]


def write_raw_score(f, rids, visits, preds, labels):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    rids = rids.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        rid = str(rids[index])
        visit = str(visits[index])
        pred = "__".join(map(str, list(pred)))
        f.write(rid+'__'+visit+'__'+pred + '__' + label + '\n')

def test_LRP(filepath, file, model_name):

    net = LRP()

    test_dataset = Dataset_Early_Fusion(filepath, file)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=10, shuffle=True,
                                                  drop_last=False)

        
    net.load_state_dict(
    torch.load('/LRP/best_LRP.pth'))
    
    net.eval()
    test_y_true = []
    test_y_pred = []

    test_loss = 0
    loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS)
    with torch.no_grad():

        for step, (img, label, rid, visit) in enumerate(test_dataloader):
            img = img.float()
            label = label.long()
            out = net(img)
            net_2 = nn.Softmax(dim=1)
            out = net_2(out)
            loss = loss_fcn(out, label)
            test_loss += loss.item()

            test_y_true, test_y_pred = UT.assemble_labels(step, test_y_true, test_y_pred, label, out)

        test_loss = test_loss / (step + 1)
        test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1] == test_y_true)) / float(len(test_y_pred))
        preds = torch.max(test_y_pred, 1)[1]
        preds = np.array(preds)
        
        cal_CI_plot_close(preds, test_y_pred, test_y_true)

        print("test_loss, test_acc:", test_loss, test_acc)


if __name__ == '__main__':

    BATCH_SIZE = 5
    EPOCHS = 150

    LR = 0.000027
    LOSS_WEIGHTS = torch.tensor([1., 1.])

    file = '/test.csv'

    model_name = 'LRP'

    filepath1 = '/ADNI/npy/'

    test_LRP(filepath1, file, model_name)










