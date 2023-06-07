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

def read_csv_complete1(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

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

    return filenames, labels, rids, visits

class VGG3D(nn.Module):
    def __init__(self, num_classes=2,
                 input_shape=(1, 110, 110, 110)):  # input: input_shape:	[num_of_filters, kernel_size] (e.g. [256, 25])
        super(VGG3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=input_shape[0],  # input height
                out_channels=8,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),

            nn.Conv3d(
                in_channels=8,  # input height
                out_channels=8,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),  # activation

            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,  # input height
                out_channels=16,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),  # activation
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),  # activation
            nn.Conv3d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=64,  # input height
                out_channels=64,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),  # activation
            nn.Conv3d(
                in_channels=64,  # input height
                out_channels=64,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        fc1_output_features = 128
        self.fc1 = nn.Sequential(
            nn.Linear(100672, 128),  # 100672是182的npy
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        fc2_output_features = 64
        self.fc2 = nn.Sequential(
            nn.Linear(fc1_output_features, fc2_output_features),
            nn.BatchNorm1d(fc2_output_features),
            nn.ReLU()
        )

        if (num_classes == 2):
            self.out = nn.Linear(fc2_output_features, 2)
            self.out_act = nn.Sigmoid()
        else:
            self.out = nn.Linear(fc2_output_features, num_classes)
            self.out_act = nn.Softmax()

    def forward(self, x, drop_prob=0.8):

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, num_filter * w * h)

        x = self.fc1(x)
        x = nn.Dropout(drop_prob)(x)
        x = self.fc2(x)
        # x = nn.Dropout(drop_prob)(x)
        prob = self.out(x)  # probability
        # 		y_hat = self.out_act(prob) # label
        # 		return y_hat, prob, x    # return x for visualization
        return prob



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


def test_Vox_CNN(filepath, file, model_name):

    net = VGG3D()

    test_dataset = Dataset_Early_Fusion(filepath, file)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=10, shuffle=True,
                                                  drop_last=False)

    net.load_state_dict(torch.load("/Vox/best_Vox.pth"))
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
            loss = loss_fcn(out, label)
            test_loss += loss.item()

            test_y_true, test_y_pred = assemble_labels(step, test_y_true, test_y_pred, label, out)

        test_loss = test_loss / (step + 1)
        test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1] == test_y_true)) / float(len(test_y_pred))

        preds = torch.max(test_y_pred, 1)[1]
        preds = np.array(preds)
        cal_CI_plot_close(preds, test_y_pred, test_y_true)

        print("test_loss, test_acc:", test_loss, test_acc)



if __name__ == '__main__':

    BATCH_SIZE = 10
    EPOCHS = 150

    LR = 0.000027
    LOSS_WEIGHTS = torch.tensor([1., 1.])

    file = '/test.csv'

    model_name = 'Vox'

    filepath1 = '/ADNI/npy/'

    test_Vox_CNN(filepath1, file, model_name)













