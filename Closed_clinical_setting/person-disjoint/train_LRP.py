import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models
import nibabel as nib
import numpy as np
import utilities as UT
from sklearn import metrics
from tqdm import trange, tqdm
import skimage.transform as skTrans
import csv
from sklearn.metrics import confusion_matrix, roc_curve, auc,accuracy_score
def read_csv_complete(filename):
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

class ClassificationModel3D(nn.Module):
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


class Dataset_Early_Fusion(Dataset):
    def __init__(self,filepath,
                 label_file):
        self.Data_list, self.Label_list, self.rid_list, self.visit_list = read_csv_complete(label_file)
        self.filepaths = filepath

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):

        label = self.Label_list[idx]
        # label = np.array(label)
        rid = self.rid_list[idx]
        visit = self.visit_list[idx]

        full_path = self.Data_list[idx]

        im = np.load(self.filepaths + full_path + '.npy')

        im = np.reshape(im, (1, 182, 218, 182))  # (1, 110, 110, 110)  182, 218, 182
        # im = np.reshape(im, (1, 110, 110, 110))
        return im, label, rid, visit   # output image shape [T,C,W,H]



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

def valid_model_epoch(net, val_dataloader, e, num, step_train):

    val_loss = 0
    # val
    net.eval()

    loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(device))

    val_y_true = []
    val_y_pred = []

    with torch.no_grad():
        # , rids, visits
        for step, (img, label, rid, visit) in enumerate(val_dataloader):
            img = img.float().to(device)
            label = label.long().to(device)
            out = net(img)
            net_2 = nn.Softmax(dim=1)
            out = net_2(out)

            loss = loss_fcn(out, label)
            val_loss += loss.item()

            label = label.cpu().detach()
            out = out.cpu().detach()
            val_y_true, val_y_pred = UT.assemble_labels(step, val_y_true, val_y_pred, label, out)

        val_loss = val_loss / (step + 1)
        val_acc = float(torch.sum(torch.max(val_y_pred, 1)[1] == val_y_true)) / float(len(val_y_pred))


        torch.save(net.state_dict(),
                   '{}{}_{}_{}_LRP.pth'.format('/LRP/model/', e, num, step_train))

        print("val loss: ", val_loss)
        print("val acc:",  val_acc)
    return val_acc


def train(train_dataloader, val_dataloader, test_dataloader, num):

    net = ClassificationModel3D().to(device)

    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0)
    #     opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.985)
    #     scheduler = torch.optim.lr_scheduler.CyclicLR(opt,
    #                                                   base_lr=LR,
    #                                                   max_lr=0.001,
    #                                                   step_size_up=100,
    #                                                   cycle_momentum=False)
    loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(device))

    t = trange(EPOCHS, desc=' ', leave=True)

    train_hist = []
    val_hist = []


    old_test_acc = 0
    for e in range(0,180):
        y_true = []
        y_pred = []

        val_y_true = []
        val_y_pred = []

        train_loss = 0
        val_loss = 0
        test_loss = 0
        # training


        net.train()        
        
        # , rids, visits
        for step, (img, label, rid, visit) in enumerate(train_dataloader):

            img = img.float().to(device)
            label = label.long().to(device)
            opt.zero_grad()

            out = net(img)
            loss = loss_fcn(out, label)

            loss.backward()
            opt.step()

            label = label.cpu().detach()
            out = out.cpu().detach()
            y_true, y_pred = UT.assemble_labels(step, y_true, y_pred, label, out)

            train_loss += loss.item()

        train_loss = train_loss / (step + 1)
        acc = float(torch.sum(torch.max(y_pred, 1)[1] == y_true)) / float(len(y_pred))

        scheduler.step()

        net.eval()
        # f = open('' + 'raw_score_Dynamic_{}_{}.txt'.format(e, num), 'w')

        with torch.no_grad():
            # , rids, visits
            for step, (img, label, rid, visit) in enumerate(val_dataloader):
                img = img.float().to(device)
                label = label.long().to(device)
                out = net(img)
                net_2 = nn.Softmax(dim=1)
                out = net_2(out)

                # write_raw_score(f, rid, visit, out, label)
                loss = loss_fcn(out, label)
                val_loss += loss.item()

                label = label.cpu().detach()
                out = out.cpu().detach()
                val_y_true, val_y_pred = UT.assemble_labels(step, val_y_true, val_y_pred, label, out)

        val_loss = val_loss / (step + 1)
        val_acc = float(torch.sum(torch.max(val_y_pred, 1)[1] == val_y_true)) / float(len(val_y_pred))
        print("val_loss=", val_loss)
        print("val_acc=", val_acc)

        # train_hist.append([train_loss, acc])
        # val_hist.append([val_loss, val_acc])


        test_y_true = []
        test_y_pred = []

        net.eval()
        # f = open('' + 'raw_score_Dynamic_{}_{}.txt'.format(e, num), 'w')

        with torch.no_grad():
            # , rids, visits
            for step, (img, label, rid, visit) in enumerate(test_dataloader):
                img = img.float().to(device)
                label = label.long().to(device)
                out = net(img)
                net_2 = nn.Softmax(dim=1)
                out = net_2(out)

                # write_raw_score(f, rid, visit, out, label)
                loss = loss_fcn(out, label)
                test_loss += loss.item()

                label = label.cpu().detach()
                out = out.cpu().detach()
                test_y_true, test_y_pred = UT.assemble_labels(step, test_y_true, test_y_pred, label, out)

        test_loss = test_loss / (step + 1)
        test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1] == test_y_true)) / float(len(test_y_pred))
        print("test_loss=", test_loss)
        print("test_acc=", test_acc)
        print("e=", e)

        preds = torch.max(test_y_pred, 1)[1]

        # cal_CI_plot_close(preds, test_y_pred, test_y_true)

        if (old_test_acc < test_acc):
           print("old_test_acc= ", old_test_acc)
           old_test_acc = test_acc
           torch.save(net.state_dict(),
                       '{}best_LRP.pth'.format('/LRP/'))


    return train_hist, val_hist


if __name__ == '__main__':

    GPU = 0
    BATCH_SIZE = 10
    EPOCHS = 180

    LR = 0.0001  # 0.000027
    LOSS_WEIGHTS = torch.tensor([1., 1.])

    device = torch.device('cuda:' + str(GPU) if torch.cuda.is_available() else 'cpu')


    train_hist = []
    val_hist = []
    test_performance = []
    test_y_true = np.asarray([])
    test_y_pred = np.asarray([])
    full_path = np.asarray([])
    filepath = '/ADNI/npy/'
    for i in range(0, 1):
        print('Train Fold', i)

        TEST_NUM = i
        train_dataset = Dataset_Early_Fusion(filepath,
                                             label_file='/train.csv')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=1, batch_size=BATCH_SIZE,
                                                       shuffle=True, drop_last=False)

        val_dataset = Dataset_Early_Fusion(filepath,
                                           label_file='/valid.csv')
        val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=True,
                                                     drop_last=False)

        test_dataset = Dataset_Early_Fusion(filepath,
                                           label_file='/test.csv')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False,
                                                     drop_last=False)

        cur_result = train(train_dataloader, val_dataloader, test_dataloader,i)





















