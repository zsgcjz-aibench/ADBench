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
# from eval_index_cal import cal_CI_plot_close

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

class Dataset_Early_Fusion(Dataset):
    def __init__(self,filepath,
                 label_file):
        self.Data_list, self.Label_list, self.rid_list, self.visit_list = read_csv_complete(label_file)
        self.filepaths = filepath

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):

        label = self.Label_list[idx]
        rid = self.rid_list[idx]
        visit = self.visit_list[idx]

        full_path = self.Data_list[idx]

        im = np.load(self.filepaths + full_path + '.npy')

        im = np.reshape(im, (1, 182, 218, 182))  # (1, 110, 110, 110)

        return im, label, rid, visit   # output image shape [T,C,W,H]


class DSA_3D_CNN(nn.Module):
    def __init__(self, num_classes=2,
                 input_shape=(1, 110, 110, 110)):  # input: input_shape:	[num_of_filters, kernel_size] (e.g. [256, 25])
        super(DSA_3D_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=input_shape[0],  # input height
                out_channels=8,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,  # input height
                out_channels=8,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,  # input height
                out_channels=8,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        self.fc1 = nn.Sequential(
            nn.Linear(104544, 2000),  # 104544是（181, 217, 181)格式的.nii， 17576是（110，110，110）的shape
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU()
        )

        if (num_classes == 2):
            self.out = nn.Linear(500, 2)
            self.out_act = nn.Sigmoid()
        else:
            self.out = nn.Linear(500, num_classes)
            self.out_act = nn.Softmax()


    def forward(self,x, drop_prob=0.8):
        x = self.conv1(x)
        # print("conv1:", x.shape)  # torch.Size([8, 8, 55, 55, 55])
        x = self.conv2(x)
        # print("conv2:",x.shape)  # torch.Size([8, 8, 27, 27, 27])
        x = self.conv3(x)
        # print("conv3:",x.shape)  # torch.Size([8, 8, 13, 13, 13])
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, num_filter * w * h)
        # print("view:",x.shape)   # torch.Size([8, 17576])
        x = self.fc1(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        x = nn.Dropout(drop_prob)(x)
        prob = self.out(x)  # probability
        # 		y_hat = self.out_act(prob) # label
        # 		return y_hat, prob, x    # return x for visualization
        return prob

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

def savemodel(net, val_dataloader, test_dataloader, e, step_strain, loss_fcn):

    old_test_acc = 0

    val_y_true = []
    val_y_pred = []

    val_loss = 0
    test_loss = 0

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

    f = open('/DSA/test_score/' + 'raw_score_Dynamic_{}_{}.txt'.format(e, step_strain), 'w')

    with torch.no_grad():
        # , rids, visits
        for step, (img, label, rid, visit) in enumerate(test_dataloader):
            img = img.float().to(device)
            label = label.long().to(device)
            out = net(img)
            net_2 = nn.Softmax(dim=1)
            out = net_2(out)

            write_raw_score(f, rid, visit, out, label)
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


    fpr, tpr, thersholds = roc_curve(y_true=test_y_true, y_score=test_y_pred[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:", roc_auc)


def train(train_dataloader, val_dataloader, test_dataloader, i):

    net = DSA_3D_CNN().to(device)

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
    pred_result = []
    old_acc = 0
    old_auc = 0
    test_acc = 0
    best_epoch = 0
    # test_performance = []
    # t：epochs
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

        val_y_true = []
        val_y_pred = []

        val_loss = 0
        test_loss = 0

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

        # f = open('/home/lxj/mxx/dataset_no_rid_repeat/DSA/test_score/' + 'raw_score_Dynamic_{.txt'.format(e,), 'w')

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

        if (old_test_acc < test_acc):
           print("old_test_acc= ", old_test_acc)
           old_test_acc = test_acc
           torch.save(net.state_dict(),
                       '{}best_DSA.pth'.format('/DSA/'))

        print("test_loss=", test_loss)
        print("test_acc=", test_acc)
        print("e=", e)

        fpr, tpr, thersholds = roc_curve(y_true=test_y_true, y_score=test_y_pred[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        print("roc_auc:", roc_auc)



    return train_hist, val_hist


if __name__ == '__main__':

    GPU = 1
    BATCH_SIZE = 5
    EPOCHS = 180

    LR = 0.000027    # 0.000027
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




















