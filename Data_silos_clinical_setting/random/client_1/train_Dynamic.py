import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models
import nibabel as nib
import cv2
import numpy as np
import utilities as UT
# from sklearn import metrics
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


class att(nn.Module):
    def __init__(self, input_channel):
        "the soft attention module"
        super(att, self).__init__()
        self.channel_in = input_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=512,
                kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=64,
                kernel_size=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        mask = x
        mask = self.conv1(mask)
        mask = self.conv2(mask)
        mask = self.conv3(mask)
        att = self.conv4(mask)
        # print(att.size())
        output = torch.mul(x, att)
        return output

class Dynamic_images_VGG(nn.Module):
    def __init__(self,
                 num_classes=2,
                 feature='Vgg11',
                 feature_shape=(512, 7, 7),
                 pretrained=True,
                 requires_grad=True):

        super(Dynamic_images_VGG, self).__init__()

        # Feature Extraction
        if (feature == 'Alex'):
            self.ft_ext = models.alexnet(pretrained=pretrained)
            self.ft_ext_modules = list(list(self.ft_ext.children())[:-2][0][:9])

        elif (feature == 'Res34'):
            self.ft_ext = models.resnet34(pretrained=pretrained)
            self.ft_ext_modules = list(self.ft_ext.children())[0:3] + list(self.ft_ext.children())[
                                                                      4:-2]  # remove the Maxpooling layer

        elif (feature == 'Res18'):
            self.ft_ext = models.resnet18(pretrained=pretrained)
            self.ft_ext_modules = list(self.ft_ext.children())[0:3] + list(self.ft_ext.children())[
                                                                      4:-2]  # remove the Maxpooling layer

        elif (feature == 'Vgg16'):
            self.ft_ext = models.vgg16(pretrained=pretrained)
            self.ft_ext_modules = list(self.ft_ext.children())[0][:30]  # remove the Maxpooling layer

        elif (feature == 'Vgg11'):
            self.ft_ext = models.vgg11(pretrained=pretrained)
            self.ft_ext_modules = list(self.ft_ext.children())[0][:19]  # remove the Maxpooling layer

        elif (feature == 'Mobile'):
            self.ft_ext = models.mobilenet_v2(pretrained=pretrained)
            self.ft_ext_modules = list(self.ft_ext.children())[0]  # remove the Maxpooling layer

        self.ft_ext = nn.Sequential(*self.ft_ext_modules)
        for p in self.ft_ext.parameters():
            p.requires_grad = requires_grad

        # Classifier
        if (feature == 'Alex'):
            feature_shape = (256, 5, 5)
        elif (feature == 'Res34'):
            feature_shape = (512, 7, 7)
        elif (feature == 'Res18'):
            feature_shape = (512, 7, 7)
        elif (feature == 'Vgg16'):
            feature_shape = (512, 6, 6)
        elif (feature == 'Vgg11'):
            feature_shape = (512, 6, 6)
        elif (feature == 'Mobile'):
            feature_shape = (1280, 4, 4)

        conv1_output_features = int(feature_shape[0])
        print("conv1_output_features:", conv1_output_features)

        fc1_input_features = int(conv1_output_features * feature_shape[1] * feature_shape[2])
        fc1_output_features = int(conv1_output_features * 2)
        fc2_output_features = int(fc1_output_features / 4)

        self.attn = att(conv1_output_features)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_shape[0],
                out_channels=conv1_output_features,
                kernel_size=1,
            ),
            nn.BatchNorm2d(conv1_output_features),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(fc1_input_features, fc1_output_features),
            nn.BatchNorm1d(fc1_output_features),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(fc1_output_features, fc2_output_features),
            nn.BatchNorm1d(fc2_output_features),
            nn.ReLU()
        )

        self.out = nn.Linear(fc2_output_features, num_classes)

    def forward(self, x, drop_prob=0.5):
        x = self.ft_ext(x)
        # print(x.size())
        # print("1:", x.shape)
        x = self.attn(x)
        # print("2:", x.shape)
        # x = self.conv1(x)
        x = x.view(x.size(0), -1)
        # print("3:", x.shape)
        x = self.fc1(x)
        # print("4:", x.shape)
        x = nn.Dropout(drop_prob)(x)
        x = self.fc2(x)
        # print("5:", x.shape)
        x = nn.Dropout(drop_prob)(x)
        # print("6:", x.shape)
        prob = self.out(x)
        return prob

class Dataset_Early_Fusion_Dynamic(Dataset):
    def __init__(self, filepath,
                 label_file):
        self.Data_list, self.Label_list, self.rid_list, self.visit_list = read_csv_complete(label_file)
        # print("self.files:", self.Data_list)
        self.filepaths = filepath

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        rid = self.rid_list[idx]
        visit = self.visit_list[idx]

        full_path = self.Data_list[idx]

        im = np.load(self.filepaths + full_path + '.npy')
        # im = nib.load(self.filepaths + full_path + '.nii').get_data()
        # im = np.array(im)
        # print("im111:", im.shape)  # (110, 110, 110)
        im = skTrans.resize(im, (110, 110, 110), order=1, preserve_range=True)
        im = np.reshape(im, (110, 110, 110, 1))  # (110, 110, 110, 1)
        # print("reshape::", im.shape)
        im = get_dynamic_image(im)  # 要4维才可以?
        # print("get_dynamic_image:", im.shape)  # (110, 110)，所有要再次进行np.expand_dims加维度
        im = np.expand_dims(im, 0)
        # print("im222:", im.shape)  # (1, 110, 110)
        im = np.concatenate([im, im, im], 0)

        return im, label, rid, visit   # output image shape [T,C,W,H]

def get_dynamic_image(frames, normalized=True):
    """ Adapted from https://github.com/tcvrick/Python-Dynamic-Images-for-Action-Recognition"""
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""

    def _get_channel_frames(iter_frames, num_channels):
        """ Takes a list of frames and returns a list of frame lists split by channel. """
        frames = [[] for channel in range(num_channels)]

        for frame in iter_frames:
            for channel_frames, channel in zip(frames, cv2.split(frame)):
                channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
        for i in range(len(frames)):
            frames[i] = np.array(frames[i])
        return frames

    def _compute_dynamic_image(frames):
        """ Adapted from https://github.com/hbilen/dynamic-image-nets """
        num_frames, h, w, depth = frames.shape

        # Compute the coefficients for the frames.
        coefficients = np.zeros(num_frames)
        for n in range(num_frames):
            cumulative_indices = np.array(range(n, num_frames)) + 1
            coefficients[n] = np.sum(((2 * cumulative_indices) - num_frames) / cumulative_indices)

        # Multiply by the frames by the coefficients and sum the result.
        x1 = np.expand_dims(frames, axis=0)
        x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
        result = x1 * x2
        return np.sum(result[0], axis=0).squeeze()

    num_channels = frames[0].shape[2]
    # print(num_channels)
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image

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

    # 一个验证集保存一个txt文本
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

        torch.save(net.state_dict(),
                   '{}{}_{}_{}_Dynamic.pth'.format('/Dynamic/model/', e, num, step_train))

        print("val loss: ", val_loss)
        print("val acc:",  val_acc)
    return val_acc


def train(train_dataloader, val_dataloader, test_dataloader,num):


    net = Dynamic_images_VGG().to(device)

    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.985)

    loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(device))

    t = trange(EPOCHS, desc=' ', leave=True)

    train_hist = []
    val_hist = []

    old_val_acc = 0
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
        # f = open('/home/mxx/LRP/dataset_rid_no_repeat/Dynamic/test_score/' + 'raw_score_Dynamic2_{}_{}.txt'.format(e, num), 'w')

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
                       '{}best_Dynamic.pth'.format('//Dynamic/'))

        

    return train_hist, val_hist


if __name__ == '__main__':

    GPU = 0
    BATCH_SIZE = 10
    EPOCHS = 180

    LR = 0.0001    # 0.000027
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


        train_dataset = Dataset_Early_Fusion_Dynamic(filepath,
                                             label_file='/train.csv')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=1, batch_size=BATCH_SIZE,
                                                       shuffle=True, drop_last=False)

        val_dataset = Dataset_Early_Fusion_Dynamic(filepath,
                                           label_file='/valid.csv')
        val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=True,
                                                     drop_last=False)

        test_dataset = Dataset_Early_Fusion_Dynamic(filepath,
                                            label_file='/test.csv')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False,
                                                      drop_last=False)

        cur_result = train(train_dataloader, val_dataloader, test_dataloader, i)


















