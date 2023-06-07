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
from NeurIPS.Data_silos_clinical_setting.random.utils.eval_index_cal import cal_CI_plot_close3, assemble_labels, cal_CI_plot_close

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

        im = np.reshape(im, (1, 182, 218, 182))

        return im, label, rid, visit

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
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, num_filter * w * h)
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


def model_test(test_dataloader):

    net = DSA_3D_CNN()

    net.load_state_dict(torch.load("/DSA/best_DSA.pth"))

    test_y_true = []
    test_y_pred = []

    net.eval()
    test_loss = 0
    loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS)

    #f = open('/DSA/' + 'raw_score_DSA_{}.txt'.format('best'),'w')

    with torch.no_grad():
        # , rids, visits
        for step, (img, label, rid, visit) in enumerate(test_dataloader):
            img = img.float()
            label = label.long()
            out = net(img)
            net_2 = nn.Softmax(dim=1)
            out = net_2(out)

            #write_raw_score(f, rid, visit, out, label)
            loss = loss_fcn(out, label)
            test_loss += loss.item()

            label = label.cpu().detach()
            out = out.cpu().detach()
            test_y_true, test_y_pred = UT.assemble_labels(step, test_y_true, test_y_pred, label, out)

    preds = torch.max(test_y_pred, 1)[1]
    preds = np.array(preds)

    cal_CI_plot_close(preds, test_y_pred, test_y_true)
    test_loss = test_loss / (step + 1)
    test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1] == test_y_true)) / float(len(test_y_pred))
    print("test_loss=", test_loss)
    print("test_acc=", test_acc)

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
    test_dataset = Dataset_Early_Fusion(filepath, label_file='./test.csv')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False,
                                                 drop_last=False)

    cur_result = model_test(test_dataloader)




















