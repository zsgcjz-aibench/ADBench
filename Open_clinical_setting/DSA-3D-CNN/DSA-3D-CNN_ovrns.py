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
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import random
import eval_index_cal as evindex
import pandas as pd
import copy as cp



import torch.nn.functional as F


#继承torch.utils.Dataset定义自己的数据方法
#要自定义自己的 Dataset 类，需要重载两个方式，【__len__】、【__getitem__】
class Dataset_Early_Fusion(Dataset):
    def __init__(self,filepath,
                 label_file):
        # self.files = UT.read_csv(label_file)
        # self.Data_list, self.Label_list = UT.read_csv_1(label_file)
        self.Data_list, self.Label_list, self.Rid_list, self.Viscode_list = UT.read_csv_1(label_file)
        # print("self.files:", self.Data_list)
        self.filepaths = filepath

    def __len__(self): #返回数据集的大小
        return len(self.Data_list)

    def __getitem__(self, idx): #实现索引数据集中的某一个元素

        label = self.Label_list[idx]

        full_path = self.Data_list[idx]

        rid = self.Rid_list[idx]

        viscode = self.Viscode_list[idx]

        # im = np.load(self.filepaths + full_path + '.npy')
        im = np.load(self.filepaths + full_path + '.npy')# (110, 110, 110)
        # print("im.shape:", im.shape)  # (110, 110, 110)
        im = skTrans.resize(im, (110, 110, 110), order=1, preserve_range=True)
        # im = np.array(im)
        im = np.reshape(im, (1, 110, 110, 110))  # (1, 110, 110, 110)

        return im, label, full_path, rid, viscode  # output image shape [T,C,W,H]


        # return im, int(label), full_path  # output image shape [T,C,W,H]


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
            nn.Linear(17576, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU()
        )

        if (num_classes == 2):
            # 未修改前
            # self.out = nn.Linear(500, 2)
            # self.out_act = nn.Sigmoid()
            # # 修改后
            self.out1 = nn.Linear(500, 1)
            self.out2 = nn.Linear(500, 1)
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

        # 修改前
        # x = self.out(x)  # probability
        # prob = self.out_act(x)
        # 修改后
        out1 = self.out1(x)
        out2 = self.out2(x)
        out3 = torch.cat((out1, out2), dim=1)
        prob = self.out_act(out3)

        # 		y_hat = self.out_act(prob) # label
        # 		return y_hat, prob, x    # return x for visualization
        return prob

def train(train_dataloader, val_dataloader):
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
    # loss_fcn = F.binary_cross_entropy(weight=LOSS_WEIGHTS.to(device))

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

        rid_list = []
        viscode_list = []

        val_y_true = []
        val_y_pred = []

        val_rid_list = []
        val_viscode_list = []

        train_loss = 0
        val_loss = 0

        # training
        net.train()
        for step, (img, label, _, rid, viscode) in enumerate(train_dataloader):
            img = img.float().to(device)
            label = label.float().to(device)
            label = F.one_hot(label.to(torch.int64), 2).float()

            rid = rid.float().to(device)
            rid = rid.cpu()
            rid = np.array(rid, dtype=int)
            rid = list(rid)
            viscode = list(viscode)

            rid_list += rid
            viscode_list += viscode

            opt.zero_grad()
            # print("img:", img.shape)  # img: torch.Size([8, 1, 110, 110, 110])

            out = net(img)
            loss = F.binary_cross_entropy(out, label)
            # loss = loss_fcn(out, label)

            loss.backward()
            opt.step()

            label = label.cpu().detach()
            out = out.cpu().detach()
            y_true, y_pred = UT.assemble_labels(step, y_true, y_pred, label, out)

            train_loss += loss.item()

        train_loss = train_loss / (step + 1)
        acc = float(torch.sum(torch.max(y_pred, 1)[1] == y_true[:, 1])) / float(len(y_pred))
        auc = metrics.roc_auc_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true[:, 1], torch.max(y_pred, 1)[1])
        precision = metrics.precision_score(y_true[:, 1], torch.max(y_pred, 1)[1])
        recall = metrics.recall_score(y_true[:, 1], torch.max(y_pred, 1)[1])
        ap = metrics.average_precision_score(y_true[:, 1], torch.max(y_pred, 1)[1])  # average_precision

        scheduler.step()

        # val
        net.eval()
        full_path = []
        with torch.no_grad():
            for step, (img, label, _, rid, viscode) in enumerate(val_dataloader):
                img = img.float().to(device)
                label = label.float().to(device)
                label = F.one_hot(label.to(torch.int64), 2).float()
                out = net(img)

                rid = rid.float().to(device)
                rid = rid.cpu()
                rid = np.array(rid, dtype=int)
                rid = list(rid)
                viscode = list(viscode)

                val_rid_list += rid
                val_viscode_list += viscode

                loss = F.binary_cross_entropy(out, label)
                # loss = loss_fcn(out, label)
                val_loss += loss.item()

                label = label.cpu().detach()
                out = out.cpu().detach()
                val_y_true, val_y_pred = UT.assemble_labels(step, val_y_true, val_y_pred, label, out)

                for item in _:
                    full_path.append(item)

        val_loss = val_loss / (step + 1)
        val_acc = float(torch.sum(torch.max(val_y_pred, 1)[1] == val_y_true[:, 1])) / float(len(val_y_pred))
        val_auc = metrics.roc_auc_score(val_y_true, val_y_pred)
        val_f1 = metrics.f1_score(val_y_true[:, 1], torch.max(val_y_pred, 1)[1])
        val_precision = metrics.precision_score(val_y_true[:, 1], torch.max(val_y_pred, 1)[1])
        val_recall = metrics.recall_score(val_y_true[:, 1], torch.max(val_y_pred, 1)[1])
        val_ap = metrics.average_precision_score(val_y_true[:, 1], torch.max(val_y_pred, 1)[1])  # average_precision

        train_hist.append([train_loss, acc, auc, f1, precision, recall, ap])
        val_hist.append([val_loss, val_acc, val_auc, val_f1, val_precision, val_recall, val_ap])

        t.set_description("Epoch: %i, train loss: %.4f, train acc: %.4f, val loss: %.4f, val acc: %.4f, test acc: %.4f"
                          % (e, train_loss, acc, val_loss, val_acc, test_acc))

        if not os.path.exists("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns"):
            os.makedirs("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns")
        torch.save(net.state_dict(),
                   "/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns/" + str(e) + "_" +'DSA_3D_CNN_normal.pth')

        if (old_acc < val_acc):
            if not os.path.exists("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns"):
                os.makedirs("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns/")
            torch.save(net.state_dict(),
                       "/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns/" + 'DSA_3D_CNN_best.pth')
            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1] == test_y_true[:, 1])) / float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true, test_y_pred)
            test_f1 = metrics.f1_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])  # average_precision

        if (old_acc == val_acc) and (old_auc < val_auc):
            if not os.path.exists("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns"):
                os.makedirs("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns/")
            torch.save(net.state_dict(),
                       "/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns/" + 'DSA_3D_CNN_best.pth')
            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1] == test_y_true[:, 1])) / float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true, test_y_pred)
            test_f1 = metrics.f1_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true[:, 1], torch.max(test_y_pred, 1)[1])  # average_precision

            test_performance = [best_epoch, test_loss, test_acc, test_auc, test_f1, test_precision, test_recall,
                                test_ap]

        # 保存ovrns的预测值
        val_y_pred = np.array(val_y_pred)
        val_y_true = np.array(val_y_true)
        save_result = []
        # save_label = []
        for index in range(0, len(val_y_true)):
            save_result.append([val_rid_list[index], val_viscode_list[index], val_y_pred[index, 0], val_y_pred[index, 1],
                                val_y_true[index, 1]])

        save_result = pd.DataFrame(save_result, columns=['RID', 'VISCODE', 'pred_cn', 'pred_ad', 'label'])
        save_result.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/ovrns/DSA_3D_CNN_close_ac_valid_preds.csv', index=0)
        # for index in range(0, len(val_y_true)):
        #     save_label.append([val_y_true[index, 0], val_y_true[index, 1]])
        #
        # save_label = pd.DataFrame(save_label)
        # save_label.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/close/DSA_3D_CNN_ac_valid_labels.csv', index=0)


    return train_hist, val_hist, test_performance, test_y_true, test_y_pred, full_path


def test(test_dataloader):
    test_hist = []
    # loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(device))
    # loss_fcn = F.binary_cross_entropy(weight=LOSS_WEIGHTS.to(device))
    test_y_true = []
    test_y_pred = []

    test_rid_list = []
    test_viscode_list = []

    test_loss = 0
    net = net = DSA_3D_CNN().to(device)
    net.load_state_dict(
        torch.load("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns/DSA_3D_CNN_best.pth"))  # ///////////
    net.eval()
    with torch.no_grad():
        for step, (img, label, _, rid, viscode) in enumerate(test_dataloader):
            img = img.float().to(device)
            label = label.long().to(device)
            out = net(img)
            # print("type(out)= ", type(out))
            # net_2 = nn.Softmax(dim=1)
            # out = net_2(out)

            label = label.cpu().detach()
            out = out.cpu().detach()

            rid = rid.float().to(device)
            rid = rid.cpu()
            rid = np.array(rid, dtype=int)
            rid = list(rid)
            viscode = list(viscode)

            test_rid_list += rid
            test_viscode_list += viscode

            test_y_true, test_y_pred = UT.assemble_labels(step, test_y_true, test_y_pred, label, out)

    preds = np.array(test_y_pred)
    labels = np.array(test_y_true)

    # evindex.cal_CI(preds, labels)
    evindex.cal_CI_plot(preds, labels)
    # evindex.cal_CI_plot_close3(preds, labels)

    # 保存ovrns的预测值
    save_result = []
    for index in range(0, len(labels)):
        save_result.append([test_rid_list[index], test_viscode_list[index], preds[index, 0], preds[index, 1],
                            test_y_true[index]])

    save_result = pd.DataFrame(save_result, columns=['RID', 'VISCODE', 's1', 's2', 'label'])
    save_result.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/ovrns/DSA_3D_CNN_open_test_preds.csv', index=0)
    # labels = pd.DataFrame(labels)
    # labels.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/close/DSA_3D_CNN_ac_test_labels.csv', index=0)
    
    

    # 保存openmax的预测值
    # save_result = []
    # for index in range(0, len(labels)):
    #     save_result.append([preds[index, 0], preds[index, 1]])
    # 
    # save_result = pd.DataFrame(save_result)
    # save_result.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/DSA_3D_CNN_ac_valid_preds.csv', index=0)
    # labels = pd.DataFrame(labels)
    # labels.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/DSA_3D_CNN_ac_valid_labels.csv', index=0)

    return


features_in_hook = []
features_out_hook = []


# 使用 hook 函数
def hook(module, fea_in, fea_out):
    fea_in_np = fea_in[0].cpu().numpy()
    for i in range(fea_in_np.shape[0]):
        features_in_hook.append(fea_in_np[i, :])  # 勾的是指定层的输入.data
    # features_in_hook.append(fea_in_np)  # 勾的是指定层的输入.data
    # 只取前向传播的数值
    fea_out_np = list(fea_out[0].cpu().numpy())
    features_out_hook.append(fea_out_np)  # 勾的是指定层的输出.data
    return None


def get_av_softmax(test_dataloader, layer_name):
    net = net = DSA_3D_CNN().to(device)
    net.load_state_dict(
        torch.load("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns/DSA_3D_CNN_best.pth"))  # ///////////
    # print(net)
    for (name, module) in net.named_modules():
        # print(name)
        if name == layer_name:
            module.register_forward_hook(hook=hook)

    # net = net.cuda(gpu)
    net.eval()
    test_rid_list = []
    test_viscode_list = []
    with torch.no_grad():
        for step, (img, label, _, rid, viscode) in enumerate(test_dataloader):
            img = img.float().to(device)
            label = label.long().to(device)
            net(img)

            label = label.cpu().detach()

            rid = rid.float().to(device)
            rid = rid.cpu()
            rid = np.array(rid, dtype=int)
            rid = list(rid)
            viscode = list(viscode)

            test_rid_list += rid
            test_viscode_list += viscode

    ridlist = np.array(test_rid_list)
    viscodelist = np.array(test_viscode_list)
    print("features_out_hook= ", np.shape(features_out_hook))
    features_in_hook_np = cp.deepcopy(features_in_hook)
    features_in_hook_np = np.array(features_in_hook_np)
    print(features_in_hook_np.shape)  # 勾的是指定层的输入

    result = pd.DataFrame(ridlist, columns=['RID'])
    result['VISCODE'] = viscodelist
    result['av1'] = features_in_hook_np[:, 0]
    result['av2'] = features_in_hook_np[:, 1]
    # print("result.shape= ", np.shape(result))

    result.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/ovrns/DSA-3D-CNN_open_test_av.csv',
                  index=0)


def get_out_layer(test_dataloader):
    net = DSA_3D_CNN().to(device)
    net.load_state_dict(torch.load('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/DSA_3D_CNN_best.pth'))
    # net.out_act = nn.Sequential([])
    test_rid_list = []
    test_viscode_list = []
    test_y_true = []
    test_y_pred = []

    out = []
    net.eval()
    with torch.no_grad():
        for step, (img, label, _, rid, viscode) in enumerate(test_dataloader):
            out = img.float().to(device)
            label = label.long().to(device)

            for name, module in net.named_children():
                out = module(out)
                if name == 'out_act':
                    out.append(out.data)
    print("out.shape= ", np.shape(out))
    #         label = label.cpu().detach()
    #         out = out.cpu().detach()
    # 
    #         rid = rid.float().to(device)
    #         rid = rid.cpu()
    #         rid = np.array(rid, dtype=int)
    #         rid = list(rid)
    #         viscode = list(viscode)
    # 
    #         test_rid_list += rid
    #         test_viscode_list += viscode
    # 
    #         test_y_true, test_y_pred = UT.assemble_labels(step, test_y_true, test_y_pred, label, out)
    # 
    # preds = np.array(test_y_pred)
    # labels = np.array(test_y_true)
    # 
    # 
    # # 保存ovrns的预测值
    # save_result = []
    # for index in range(0, len(labels)):
    #     save_result.append([test_rid_list[index], test_viscode_list[index], preds[index, 0], preds[index, 1]])
    # 
    # save_result = pd.DataFrame(save_result, columns=['RID', 'VISCODE', 'o1', 'o2'])
    # save_result.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/openmax/DSA_3D_CNN_open_test_av.csv',
    #                    index=0)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        activation[name] = output.detach()
    return hook

def new_get_layer_output(test_dataloader):
    net = DSA_3D_CNN().to(device)
    net.load_state_dict(torch.load('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrns/DSA_3D_CNN_best.pth'))
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

            rid = rid.float().to(device)
            rid = rid.cpu()
            rid = np.array(rid, dtype=int)
            rid = list(rid)
            viscode = list(viscode)

            test_rid_list += rid
            test_viscode_list += viscode
            
    activation_list = np.array(activation_list.cpu())
    print("activation_list.shape= ", np.shape(activation_list))

    # 保存ovrns的预测值
    save_result = []
    for index in range(0, len(test_rid_list)):
        save_result.append([test_rid_list[index], test_viscode_list[index], activation_list[index, 0], activation_list[index, 1]])

    save_result = pd.DataFrame(save_result, columns=['RID', 'VISCODE', 'o1', 'o2'])
    save_result.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/ovrns/DSA_3D_CNN_ac_valid_av.csv',
                       index=0)
    
            



if __name__ == '__main__':
    GPU = 1
    BATCH_SIZE = 4
    EPOCHS = 100

    LR = 0.000015
    LOSS_WEIGHTS = torch.tensor([1., 1.])

    device = torch.device('cuda:' + str(GPU) if torch.cuda.is_available() else 'cpu')

    train_hist = []
    val_hist = []
    test_performance = []
    test_y_true = np.asarray([])
    test_y_pred = np.asarray([])
    full_path = np.asarray([])

    filepath = '/home/lxs/ncomms2022-main--datastore/MRI_process/test_data/processed/npy/'
    trainlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/train.csv'
    validlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/valid.csv'
    testlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/test.csv'
    open_testlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/open_test.csv'

    #
    # for i in range(0, 5):
    #     print('Train Fold', i)
    #
    #     TEST_NUM = i
    #     # TRAIN_LABEL, TEST_LABEL = prep_data(LABEL_PATH, TEST_NUM)
    #
    #     train_dataset = Dataset_Early_Fusion(filepath, label_file=trainlabel_file)
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=1, batch_size=BATCH_SIZE,
    #                                                    shuffle=True, drop_last=True)
    #
    #     val_dataset = Dataset_Early_Fusion(filepath, label_file=validlabel_file)
    #     val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False,
    #                                                  drop_last=False)
    #
    #     cur_result = train(train_dataloader, val_dataloader)
    #
    #     train_hist.append(cur_result[0])
    #     val_hist.append(cur_result[1])

    # -------------


    print("-----------------test start--------------")
    test_dataset = Dataset_Early_Fusion(filepath=filepath, label_file=open_testlabel_file)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False,
                                                      drop_last=False)
    test(test_dataloader)

    print("------------------test end---------------")

    print("-----------------test start--------------")
    layer_name = 'out_act'
    get_av_softmax(test_dataloader, layer_name)
    # get_out_layer(test_dataloader)
    # new_get_layer_output(test_dataloader)
    print("------------------test end---------------")