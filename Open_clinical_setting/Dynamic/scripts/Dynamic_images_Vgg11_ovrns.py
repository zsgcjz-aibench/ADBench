# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models
import nibabel as nib
import skimage.transform as skTrans
import os
import numpy as np
from sklearn import metrics
from tqdm import trange, tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import utilities as UT
from ranksvm import get_dynamic_image
from sklearn.metrics import roc_curve, auc
import random
import eval_index_cal as evindex
import copy as cp
import pandas as pd

import torch.nn.functional as F

class Dataset_Early_Fusion(Dataset):
    def __init__(self,filepath,
                 label_file):
        self.ridlist, self.viscode_list, self.Data_list, self.Label_list = UT.read_csv_2(label_file)
        # print("self.files:", self.Data_list)
        self.filepaths = filepath

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        rid = self.ridlist[idx]
        viscode = self.viscode_list[idx]

        label = self.Label_list[idx]

        full_path = self.Data_list[idx]

        im = np.load(self.filepaths + full_path + '.npy')
        im = skTrans.resize(im, (110, 110, 110), order=1, preserve_range=True)
        im = np.reshape(im, (110, 110, 110, 1))  # (1, 110, 110, 110)
        im = get_dynamic_image(im)   
        im = np.expand_dims(im, 0)
        im = np.concatenate([im, im, im], 0)

        return im, label, full_path, rid, viscode  # output image shape [T,C,W,H]


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
        #print("conv1_mask:", x.shape)
        mask = self.conv1(mask)
        mask = self.conv2(mask)
        mask = self.conv3(mask)
        att = self.conv4(mask)
        # print(att.size())
        output = torch.mul(x, att)
        return output


class CNN(nn.Module):
    def __init__(self,
                 num_classes=2,
                 feature='Vgg11',
                 feature_shape=(512, 7, 7),
                 pretrained=True,
                 requires_grad=True):

        super(CNN, self).__init__()

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
        #print("conv1_output_features: ", conv1_output_features)

        fc1_input_features = int(conv1_output_features * feature_shape[1] * feature_shape[2])
        #print("fc1_input_features: ", fc1_input_features)
        fc1_output_features = int(conv1_output_features * 2)
        #print("fc1_output_features: ", fc1_output_features)
        fc2_output_features = int(fc1_output_features / 4)
        #print("fc2_output_features: ", fc2_output_features)

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

        #close setting
        # self.out = nn.Linear(fc2_output_features, num_classes)
        #ovrns setting
        self.out1 = nn.Linear(fc2_output_features, 1)
        self.out2 = nn.Linear(fc2_output_features, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x, drop_prob=0.5):
        x = self.ft_ext(x)
        # print(x.size())
        x = self.attn(x)
        # x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.Dropout(drop_prob)(x)
        x = self.fc2(x)
        x = nn.Dropout(drop_prob)(x)
        #close setting
        # x = self.out(x)
        # prob = self.out_act(x)
        #ovrns setting
        out1 = self.out1(x)
        out2 = self.out2(x)
        out3 = torch.cat((out1, out2), dim=1)
        prob = self.out_act(out3)

        prob = prob.to(torch.float)
        # out3 = out3.to(torch.float)

        # return prob, out3
        return prob


def train(train_dataloader, val_dataloader, feature='Vgg11'):
    net = CNN(feature=feature).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)
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
    test_performance = []
    for e in t:
        y_true = []
        y_pred = []
        y_pred_no_sig = []
        y_true1 = []
        ridlist = []
        viscodelist = []

        val_y_true = []
        val_y_pred = []
        val_y_no_sig = []
        val_ridlist = []
        val_viscodelist = []

        train_loss = 0
        val_loss = 0

        # training
        net.train()
        # for step, (img, label, _) in enumerate(train_dataloader):
        for step, (img, label, _, rid, viscode) in enumerate(train_dataloader):
            img = img.float().to(device)
            label = label.long().to(device)
            label = F.one_hot(label.to(torch.int64), num_classes=2).float()

            rid = rid.long().to(device)
            viscode = np.array(viscode, dtype=str)
            rid = rid.cpu().detach()
            rid = np.array(rid)

            opt.zero_grad()
            out = net(img)
            # out, no_sig = net(img)

            # loss = loss_fcn(out, label)
            loss = F.binary_cross_entropy(out, label)

            loss.backward()
            opt.step()

            label = label.cpu().detach()
            out = out.cpu().detach()
            # no_sig = no_sig.cpu().detach()

            y_true, y_pred = UT.assemble_labels(step, y_true, y_pred, label, out)
            # y_true1, y_pred_no_sig = UT.assemble_labels(step, y_true1, y_pred_no_sig, label, no_sig)

            train_loss += loss.item()

            for ridline in rid:
                ridlist.append(ridline)
            for viscodeline in viscode:
                viscodelist.append(viscodeline)

        train_loss = train_loss / (step + 1)
        acc = float(torch.sum(torch.max(y_pred, 1)[1] == y_true[:, 1])) / float(len(y_pred))
        auc = metrics.roc_auc_score(y_true[:, 1], y_pred[:, 1])
        f1 = metrics.f1_score(y_true[:, 1], torch.max(y_pred, 1)[1])
        precision = metrics.precision_score(y_true[:, 1], torch.max(y_pred, 1)[1])
        recall = metrics.recall_score(y_true[:, 1], torch.max(y_pred, 1)[1])
        ap = metrics.average_precision_score(y_true[:, 1], torch.max(y_pred, 1)[1])  # average_precision

        scheduler.step()

        # save predict 
        train_y_pred = np.array(y_pred)
        train_y_true = np.array(y_true)


        save_result = []
        # save_label = []
        for index in range(0, len(train_y_true)):

            save_result.append([ridlist[index], viscodelist[index], train_y_pred[index, 0], train_y_pred[index, 1],
                                train_y_true[index, 0], train_y_true[index, 1]])

        save_result = pd.DataFrame(save_result)
        save_path = './scripts/modelsave/ovrns/epoch_' + str(
            e) + '_Dynamic-images-VGG_ac_train_result.csv'

        save_result.to_csv(save_path, index=0,
                           header=['rid', 'viscode', 'cn_pred', 'ad_pred','cn_label', 'ad_label'])



        # val
        net.eval()
        full_path = []
        y_pred_no_sig = []
        y_true1 = []
        with torch.no_grad():
            # for step, (img, label, _) in enumerate(val_dataloader):
            for step, (img, label, _, rid, viscode) in enumerate(val_dataloader):
                img = img.float().to(device)
                label = label.long().to(device)
                label = F.one_hot(label.to(torch.int64), num_classes=2).float()

                rid = rid.long().to(device)
                viscode = np.array(viscode, dtype=str)
                rid = rid.cpu().detach()
                rid = np.array(rid)

                out = net(img)
                # out, no_sig = net(img)

                # loss = loss_fcn(out, label)
                loss = F.binary_cross_entropy(out, label)

                val_loss += loss.item()

                label = label.cpu().detach()
                out = out.cpu().detach()
                # no_sig = no_sig.cpu().detach()

                val_y_true, val_y_pred = UT.assemble_labels(step, val_y_true, val_y_pred, label, out)
                # y_true1, y_pred_no_sig = UT.assemble_labels(step, y_true1, y_pred_no_sig, label, no_sig)

                for item in _:
                    full_path.append(item)

                for ridline in rid:
                    val_ridlist.append(ridline)
                for viscodeline in viscode:
                    val_viscodelist.append(viscodeline)

        val_loss = val_loss / (step + 1)
        val_acc = float(torch.sum(torch.max(val_y_pred, 1)[1] == val_y_true[:, 1])) / float(len(val_y_pred))
        val_auc = metrics.roc_auc_score(val_y_true[:, 1], val_y_pred[:, 1])
        val_f1 = metrics.f1_score(val_y_true[:, 1], torch.max(val_y_pred, 1)[1])
        val_precision = metrics.precision_score(val_y_true[:, 1], torch.max(val_y_pred, 1)[1])
        val_recall = metrics.recall_score(val_y_true[:, 1], torch.max(val_y_pred, 1)[1])
        val_ap = metrics.average_precision_score(val_y_true[:, 1], torch.max(val_y_pred, 1)[1])  # average_precision

        train_hist.append([train_loss, acc, auc, f1, precision, recall, ap])
        val_hist.append([val_loss, val_acc, val_auc, val_f1, val_precision, val_recall, val_ap])

        t.set_description("Epoch: %i, train loss: %.4f, train acc: %.4f, val loss: %.4f, val acc: %.4f, test acc: %.4f"
                          % (e, train_loss, acc, val_loss, val_acc, test_acc))

        if not os.path.exists("./scripts/modelsave/ovrns"):
            os.makedirs("./scripts/modelsave/ovrns")
        torch.save(net.state_dict(),
                   "./scripts/modelsave/ovrns/" + str(e) + "_" +'Dynamic-images-VGG_normal.pth')
        
        if (old_acc < val_acc):

            if not os.path.exists("./scripts/modelsave/ovrns"):
                os.makedirs("./scripts/modelsave/ovrns")
            torch.save(net.state_dict(),
                       "./scripts/modelsave/ovrns/" + 'Dynamic-images-VGG_best.pth')

            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1] == test_y_true[:, 1])) / float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true[:, 1], test_y_pred[:, 1])
            test_f1 = metrics.f1_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])  # average_precision

            test_performance = [best_epoch, test_loss, test_acc, test_auc, test_f1, test_precision, test_recall,
                                test_ap]

        if (old_acc == val_acc) and (old_auc < val_auc):
            
            if not os.path.exists("./scripts/modelsave/ovrns"):
                os.makedirs("./scripts/modelsave/ovrns")
            torch.save(net.state_dict(),
                       "./scripts/modelsave/ovrns/" + 'Dynamic_images_VGG_best.pth')
            
            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1] == test_y_true[:, 1])) / float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true[:, 1], test_y_pred[:, 1])
            test_f1 = metrics.f1_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])  # average_precision

            test_performance = [best_epoch, test_loss, test_acc, test_auc, test_f1, test_precision, test_recall,
                                test_ap]

        # save predict
        val_y_pred = np.array(val_y_pred)
        val_y_true = np.array(val_y_true)

        save_result = []

        for index in range(0, len(val_y_true)):

            save_result.append([val_ridlist[index], val_viscodelist[index], val_y_pred[index, 0], val_y_pred[index, 1],
                                val_y_true[index, 0], val_y_true[index, 1]])

        save_result = pd.DataFrame(save_result)
        save_path = './scripts/modelsave/ovrns/epoch_' + str(e) + '_Dynamic-images-VGG_ac_valid_result.csv'

        save_result.to_csv(save_path, index=0,
                           header=['rid', 'viscode', 'cn_pred', 'ad_pred', 'cn_label', 'ad_label'])
    return train_hist, val_hist, test_performance, test_y_true, test_y_pred, full_path


def test(test_dataloader,feature='Vgg11'):
    test_hist = []

    test_y_true = []
    test_y_pred = []
    test_ridlist = []
    test_viscodelist = []
    y_pred_no_sig = []
    y_true1 = []

    test_loss = 0
    net = CNN(feature=feature).to(device)
    net.load_state_dict(torch.load("./scripts/modelsave/ovrns/Dynamic-images-VGG_best.pth"))                     #///////////
    net.eval()
    with torch.no_grad():
        # for step, (img, label, _) in enumerate(test_dataloader):
        for step, (img, label, _, rid, viscode) in enumerate(test_dataloader):
            img = img.float().to(device)
            label = label.long().to(device)
            # label = F.one_hot(label.to(torch.int64), 3).float()

            rid = rid.long().to(device)
            viscode = np.array(viscode, dtype=str)
            rid = rid.cpu().detach()
            rid = np.array(rid)

            out = net(img)

            label = label.cpu().detach()
            out = out.cpu().detach()

            test_y_true, test_y_pred = UT.assemble_labels(step, test_y_true, test_y_pred, label, out)

            for ridline in rid:
                test_ridlist.append(ridline)
            for viscodeline in viscode:
                test_viscodelist.append(viscodeline)


    preds = np.array(test_y_pred)
    labels = np.array(test_y_true)

    save_result = []

    for index in range(0, len(test_y_true)):
        save_result.append([test_ridlist[index], test_viscodelist[index], preds[index, 0], preds[index, 1],
                            labels[index]])

    save_result = pd.DataFrame(save_result)
    save_path = './close_train_valid_preds/ovrns/Dynamic-images-VGG_ac_valid_preds.csv'

    save_result.to_csv(save_path, index=0,
                       header=['rid', 'viscode', 's1', 's2', 'label'])


    # evindex.cal_CI_plot(preds, labels) #open setting
    # evindex.cal_CI_plot_close3(preds, labels) #close setting


features_in_hook = []
features_out_hook = []


# Using hook functions
def hook(module, fea_in, fea_out):
    fea_in_np = fea_in[0].cpu().numpy()
    # print("fea_in_np.shape= ", fea_in_np.shape)
    for i in range(fea_in_np.shape[0]):
        features_in_hook.append(fea_in_np[i, :])

    fea_out_np = list(fea_out[0].cpu().numpy())
    features_out_hook.append(fea_out_np)
    
    return None


def get_av_softmax(test_dataloader, layer_name, feature='Vgg11'):
    net = CNN(feature=feature).to(device)
    net.load_state_dict(torch.load("./scripts/modelsave/ovrns/Dynamic-images-VGG_best.pth"))
    print(net)
    for (name, module) in net.named_modules():
        # print(name)
        if name == layer_name:
            module.register_forward_hook(hook=hook)

    # net = net.cuda(gpu)
    net.eval()
    test_ridlist = []
    test_viscodelist = []
    with torch.no_grad():
        for step, (img, label, _, rid, viscode) in enumerate(test_dataloader):
            img = img.float().to(device)
            label = label.long().to(device)

            rid = rid.long().to(device)
            viscode = np.array(viscode, dtype=str)
            rid = rid.cpu().detach()
            rid = np.array(rid)

            net(img)

            for ridline in rid:
                test_ridlist.append(ridline)
            for viscodeline in viscode:
                test_viscodelist.append(viscodeline)

    ridlist = np.array(test_ridlist)
    viscodelist = np.array(test_viscodelist)
    features_in_hook_np = cp.deepcopy(features_in_hook)
    features_in_hook_np = np.array(features_in_hook_np)
    print(features_in_hook_np.shape)  # 勾的是指定层的输入

    result = pd.DataFrame(ridlist, columns=['RID'])
    result['VISCODE'] = viscodelist
    result['av1'] = features_in_hook_np[:, 0]
    result['av2'] = features_in_hook_np[:, 1]
    # print("result.shape= ", np.shape(result))

    result.to_csv('./close_train_valid_preds/ovrns/Dynamic-images-VGG_ac_valid_av.csv', index=0)


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def new_get_layer_output(test_dataloader, feature='Vgg11'):
    net = CNN(feature=feature).to(device)
    net.load_state_dict(torch.load('./scripts/modelsave/ovrns/Dynamic-images-VGG_best.pth'))
    net.out.register_forward_hook(get_activation('out'))
    net.eval()
    out_list = []
    activation_list = []
    test_rid_list = []
    test_viscode_list = []
    with torch.no_grad():
        for step, (img, label, _, rid, viscode) in enumerate(test_dataloader):
            img = img.float().to(device)
            label = label.long().to(device)
            # net(img)
            output = net(img)

            label = label.cpu().detach()
            output = output.cpu().detach()
            av_value = activation['out'].to(device)
            activation_list, out_list = UT.assemble_labels(step, activation_list, out_list, av_value, output)

            rid = rid.long().to(device)
            rid = rid.cpu().detach()
            rid = np.array(rid, dtype=int)
            rid = list(rid)
            viscode = list(viscode)

            test_rid_list += rid
            test_viscode_list += viscode

    activation_list = np.array(activation_list.cpu())
    print("activation_list.shape= ", np.shape(activation_list))


    save_result = []
    for index in range(0, len(test_rid_list)):
        save_result.append(
            [test_rid_list[index], test_viscode_list[index], activation_list[index, 0], activation_list[index, 1]])

    save_result = pd.DataFrame(save_result, columns=['RID', 'VISCODE', 'o1', 'o2'])
    save_result.to_csv('./close_train_valid_preds/ovrns/Dynamic-images-VGG_ac_test_av.csv',
                       index=0)



if __name__ == '__main__':

    LABEL_PATH = 'lookcsv'
    trainlabel_file = './adni_dl/Preprocessed/ADNI2_MRI/train.csv'
    validlabel_file = './adni_dl/Preprocessed/ADNI2_MRI/valid.csv'
    testlabel_file = './adni_dl/Preprocessed/ADNI2_MRI/test.csv'
    open_testlabel_file = './adni_dl/Preprocessed/ADNI2_MRI/open_test.csv'

    GPU = 2
    BATCH_SIZE = 4
    EPOCHS = 150

    LR = 0.0001
    LOSS_WEIGHTS = torch.tensor([1., 1.])

    device = torch.device('cuda:' + str(GPU) if torch.cuda.is_available() else 'cpu')


    train_hist = []
    val_hist = []
    test_hint = []
    test_performance = []
    test_y_true = np.asarray([])
    test_y_pred = np.asarray([])
    full_path = np.asarray([])

    filepath = './MRI_process/test_data/processed/npy/' #Path for storing npy


    for i in range(0, 1):
        print('Train Fold', i)

        TEST_NUM = i

        train_dataset = Dataset_Early_Fusion(filepath, label_file=trainlabel_file)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=1, batch_size=BATCH_SIZE,
                                                       shuffle=True, drop_last=True)

        val_dataset = Dataset_Early_Fusion(filepath, label_file=validlabel_file)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False,
                                                     drop_last=False)

        cur_result = train(train_dataloader, val_dataloader)




    print("----------------- start--------------")
    test_dataset = Dataset_Early_Fusion(filepath=filepath, label_file=trainlabel_file)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False,
                                                  drop_last=True)
    test(test_dataloader)
    print("------------------test end---------------")

    # print("----------------- start--------------")
    layer_name = 'out_act'
    get_av_softmax(test_dataloader, layer_name)

   













