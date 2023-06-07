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
from sklearn.metrics import confusion_matrix, roc_curve, auc
import nibabel as nib
from NeurIPS.Data_silos_clinical_setting.random.utils.eval_index_cal import cal_CI_plot_close3, assemble_labels, cal_CI_plot_close


class Dataset_Early_Fusion_Dynamic(Dataset):
    def __init__(self, filepath,
                 label_file):
        self.Data_list, self.Label_list, self.rid_list, self.visit_list = read_csv_complete1(label_file)
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
        im = skTrans.resize(im, (110, 110, 110), order=1, preserve_range=True)
        im = np.reshape(im, (110, 110, 110, 1))
        im = get_dynamic_image(im)
        im = np.expand_dims(im, 0)
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

        # print("###:", len(rids))
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
        # print("conv1_output_features:", conv1_output_features)

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

def test_Dynamic(filepath, file, model_name):
    # print()
    # Dynamic_images_VGG
    net = Dynamic_images_VGG()

    test_dataset = Dataset_Early_Fusion_Dynamic(filepath, file)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=10, shuffle=False,
                                                      drop_last=False)

    net.load_state_dict(torch.load("/best_Dynamic.pth"))

    net.eval()
    test_y_true = []
    test_y_pred = []

    test_loss = 0
    loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS)
    with torch.no_grad():

        for step, (img, label,rid, visit) in enumerate(test_dataloader):
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
        fpr, tpr, thersholds = roc_curve(test_y_true, preds, pos_label=1)
        cal_CI_plot_close(preds, test_y_pred, test_y_true)
        roc_auc = auc(fpr, tpr)
        print("roc_auc:", roc_auc)

        print("test_loss, test_acc:", test_loss, test_acc)

if __name__ == '__main__':

    BATCH_SIZE = 5
    EPOCHS = 150

    LR = 0.000027
    LOSS_WEIGHTS = torch.tensor([1., 1.])

    file = './test.csv'

    model_name = 'Dynamic'

    filepath1 = '/ADNI/npy/'

    test_Dynamic(filepath1, file, model_name)













