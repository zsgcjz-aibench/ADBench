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
from NeurIPS.Data_silos_clinical_setting.random.utils.eval_index_cal import cal_CI_plot_close3, assemble_labels, cal_CI_plot_close

def read_csv_complete(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    filenames, labels, demors, rids, visits = [], [], [], [], []
    for line in your_list:
        try:
            age = float(line[0])
            mmse = float(line[3])
            gender = [0, 1] if line[5] == 'Female' else [1, 0]
            demor = [(age-70.0)/10.0] + gender + [(mmse-27)/2]

        except:
            continue

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
        demors.append(demor)

    return filenames, labels, demors, rids, visits


def get_AD_risk(raw):
    x1, x2 = raw[0, :, :, :], raw[1, :, :, :]
    risk = np.exp(x2) / (np.exp(x1) + np.exp(x2))
    return risk

class MLP_Data(Dataset):
    def __init__(self, Data_dir, file, roi_threshold, roi_count, choice, seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        self.roi_threshold = roi_threshold
        self.roi_count = roi_count
        if choice == 'count':
            self.select_roi_count()
        else:
            self.select_roi_thres()


        self.path = file
        self.Data_list, self.Label_list, self.demor_list, self.rid_list, self.visit_list = read_csv_complete(self.path)

        self.risk_list = [get_AD_risk(np.load(Data_dir + filename + '.npy'))[self.roi] for filename in self.Data_list]
        self.in_size = self.risk_list[0].shape[0]

    def select_roi_thres(self):
        self.roi = np.load('/DPMs/fcn_exp/test_MCC.npy')
        self.roi = self.roi > self.roi_threshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i % 3 != 0 or j % 2 != 0 or k % 3 != 0:
                        self.roi[i, j, k] = False

    def select_roi_count(self):
        self.roi = np.load('/DPMs/fcn_exp/test_MCC.npy')
        tmp = []
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i % 3 != 0 or j % 2 != 0 or k % 3 != 0: continue
                    tmp.append((self.roi[i, j, k], i, j, k))
        tmp.sort()
        tmp = tmp[-self.roi_count:]
        self.roi = self.roi != self.roi
        for _, i, j, k in tmp:
            self.roi[i, j, k] = True

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        risk = self.risk_list[idx]
        demor = self.demor_list[idx]
        rid = self.rid_list[idx]
        visit = self.visit_list[idx]
        return risk, label, np.asarray(demor).astype(np.float32), rid, visit

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.Label_list)), float(self.Label_list.count(0)), float(
            self.Label_list.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.Label_list]
        return weights, count0 / count1


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

class _MLP_C(nn.Module):
    "MLP that use DPMs from fcn and age, gender and MMSE"
    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP_C, self).__init__()
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 2)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()

    def forward(self, X1, X2):
        X = torch.cat((X1, X2), 1)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out


class _MLP_A(nn.Module):
    "MLP that only use DPMs from fcn"

    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP_A, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(fil_num)
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 2)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()

    def forward(self, X):
        X = self.bn1(X)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out


class _MLP_B(nn.Module):
    "MLP that only use age gender MMSE"

    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP_B, self).__init__()
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 2)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()

    def forward(self, X):
        out = self.do1(X)
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out


def test_MLP_C(filepath, file, fil_num, drop_rate, model_name):



    net = _MLP_A(in_size=200, fil_num=fil_num, drop_rate=drop_rate)

    test_dataset = MLP_Data(filepath, file, 0.6, 200, "count")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=10, shuffle=True,
                                                  drop_last=False)

    net.load_state_dict(torch.load("/checkpoint_dir/mlp_A/mlp_A_15.pth"))


    net.eval()
    test_y_true = []
    test_y_pred = []

    test_loss = 0
    loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS)
    with torch.no_grad():

        for step, (img, label, demors, rid, visit) in enumerate(test_dataloader):
            img = img.float()
            label = label.long()

            out = net(img)
            net_2 = nn.Softmax(dim=1)
            out = net_2(out)

            # write_raw_score(f, rid, visit, out, label)

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

    filepath = '/DPMs/fcn_exp/'

    file = '/test.csv'
    model_name = 'MPL_A'
    test_MLP_C(filepath, file, 100, 0.5, model_name)













