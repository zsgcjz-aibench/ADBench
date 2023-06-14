import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models
import nibabel as nib
import numpy as np
import pandas as pd
import utilities as UT
from sklearn import metrics
from tqdm import trange, tqdm
import skimage.transform as skTrans
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import random
import eval_index_cal as evindex
import copy


class Dataset_Early_Fusion(Dataset):
    def __init__(self,filepath,
                 label_file):
        # self.files = UT.read_csv(label_file)
        # self.ridlist, self.viscode_list, self.Data_list, self.Label_list = UT.read_csv_2(label_file)
        self.Data_list, self.Label_list, self.Rid_list, self.Viscode_list = UT.read_csv_1(label_file)
        # print("self.files:", self.Data_list)
        self.filepaths = filepath

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
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
        # prob = self.out(x)  # probability
        # prob = self.out_act(prob)
        # 修改后
        out1 = self.out1(x)
        out2 = self.out2(x)
        out3 = torch.cat((out1, out2), dim=1)
        prob = self.out_act(out3)
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
            label = label.long().to(device)

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
            # loss = loss_fcn(out, label)
            loss = F.binary_cross_entropy(out, label)

            loss.backward()
            opt.step()

            label = label.cpu().detach()
            out = out.cpu().detach()
            y_true, y_pred = UT.assemble_labels(step, y_true, y_pred, label, out)

            train_loss += loss.item()

        train_loss = train_loss / (step + 1)
        acc = float(torch.sum(torch.max(y_pred, 1)[1] == y_true)) / float(len(y_pred))
        auc = metrics.roc_auc_score(y_true, y_pred[:, 1])
        f1 = metrics.f1_score(y_true, torch.max(y_pred, 1)[1])
        precision = metrics.precision_score(y_true, torch.max(y_pred, 1)[1])
        recall = metrics.recall_score(y_true, torch.max(y_pred, 1)[1])
        ap = metrics.average_precision_score(y_true, torch.max(y_pred, 1)[1])  # average_precision

        scheduler.step()

        # val
        net.eval()
        full_path = []
        with torch.no_grad():
            for step, (img, label, _, rid, viscode) in enumerate(val_dataloader):
                img = img.float().to(device)
                label = label.long().to(device)
                out = net(img)

                rid = rid.float().to(device)
                rid = rid.cpu()
                rid = np.array(rid, dtype=int)
                rid = list(rid)
                viscode = list(viscode)

                val_rid_list += rid
                val_viscode_list += viscode

                # loss = loss_fcn(out, label)
                loss = F.binary_cross_entropy(out, label)
                val_loss += loss.item()

                label = label.cpu().detach()
                out = out.cpu().detach()
                val_y_true, val_y_pred = UT.assemble_labels(step, val_y_true, val_y_pred, label, out)

                for item in _:
                    full_path.append(item)

        val_loss = val_loss / (step + 1)
        val_acc = float(torch.sum(torch.max(val_y_pred, 1)[1] == val_y_true)) / float(len(val_y_pred))
        val_auc = metrics.roc_auc_score(val_y_true, val_y_pred[:, 1])
        val_f1 = metrics.f1_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_precision = metrics.precision_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_recall = metrics.recall_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_ap = metrics.average_precision_score(val_y_true, torch.max(val_y_pred, 1)[1])  # average_precision

        train_hist.append([train_loss, acc, auc, f1, precision, recall, ap])
        val_hist.append([val_loss, val_acc, val_auc, val_f1, val_precision, val_recall, val_ap])

        t.set_description("Epoch: %i, train loss: %.4f, train acc: %.4f, val loss: %.4f, val acc: %.4f, test acc: %.4f"
                          % (e, train_loss, acc, val_loss, val_acc, test_acc))

        if not os.path.exists("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrn/"):
            os.makedirs("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrn/")
        torch.save(net.state_dict(),
                   "/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrn/" + str(e) + "_" +'DSA_3D_CNN_normal.pth')

        if (old_acc < val_acc):
            if not os.path.exists("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrn/"):
                os.makedirs("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrn/")
            torch.save(net.state_dict(),
                       "/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/ovrn/" + 'DSA_3D_CNN_best.pth')
            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1] == test_y_true)) / float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:, 1])
            test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1])  # average_precision

        if (old_acc == val_acc) and (old_auc < val_auc):
            if not os.path.exists("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave"):
                os.makedirs("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave")
            torch.save(net.state_dict(),
                       "/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/" + 'DSA_3D_CNN_best.pth')
            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1] == test_y_true)) / float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:, 1])
            test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1])  # average_precision

            test_performance = [best_epoch, test_loss, test_acc, test_auc, test_f1, test_precision, test_recall,
                                test_ap]

        # 保存ovrns的预测值
        val_y_pred = np.array(val_y_pred)
        val_y_true = np.array(val_y_true)
        save_result = []
        # save_label = []
        for index in range(0, len(val_y_true)):
            save_result.append(
                [val_rid_list[index], val_viscode_list[index], val_y_pred[index, 0], val_y_pred[index, 1],
                    val_y_true[index, 1]])

        save_result = pd.DataFrame(save_result, columns=['RID', 'VISCODE', 'pred_cn', 'pred_ad', 'label'])
        save_result.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/ovrns/DSA_3D_CNN_ovrns_ac_valid_preds.csv',
                index=0)
    return train_hist, val_hist, test_performance, test_y_true, test_y_pred, full_path


def test(test_dataloader):
    test_hist = []
    # loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(device))
    test_y_true = []
    test_y_pred = []

    test_rid_list = []
    test_viscode_list = []

    test_loss = 0
    net = DSA_3D_CNN().to(device)
    net.load_state_dict(
        torch.load("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/DSA_3D_CNN_best.pth"))  # ///////////
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
    # 保存ovrns的预测值
    save_result = []
    for index in range(0, len(labels)):
        save_result.append([test_rid_list[index], test_viscode_list[index], preds[index, 0], preds[index, 1],
                            test_y_true[index, 1]])

    save_result = pd.DataFrame(save_result, columns=['RID', 'VISCODE', 'pred_cn', 'pred_ad', 'label'])
    save_result.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/close_train_valid_preds/ovrns/DSA_3D_CNN_ovrns_ac_test_preds.csv', index=0)


#openmax
# 加载模型  Test
# 加载数据
# def weibull_fit(model_path,score, prob, y):
#     if os.path.exists(model_path):
#         with open(model_path, 'rb') as f:
#             return pickle.load(f)
#     print('Model do not exists !      ',model_path)
#     predicted_y = np.argmax(prob, axis=1)
#     print('------------------------')
#     print(len(predicted_y))
#     print('------------------------')
# 
#     labels = np.unique(y)  # 去除重复的元素
#     av_map = {}
# 
#     for label in labels:
#         av_map[label] = score[(y == label) & (predicted_y == y), :]
#     print(len(av_map[0]))
#     print(len(av_map[1]))
# 
#     print(av_map[1])
# 
#     model = weibull_fit_tails(av_map, tail_size=300)
#     with open(model_path, 'wb') as f:
#         pickle.dump(model, f)
#     return model
# 
# 
# def weibull_fit_tails(av_map, tail_size=2000, metric_type='cosine'):
#     weibull_model = {}
#     labels = av_map.keys()
# 
#     for label in labels:
#         print(f'EVT fitting for label {label}')
#         weibull_model[label] = {}
# 
#         class_av = av_map[label]
#         class_mav = np.mean(class_av, axis=0, keepdims=True)
# 
#         av_distance = np.zeros((1, class_av.shape[0]))
#         for i in range(class_av.shape[0]):
#             av_distance[0, i] = compute_distance(class_av[i, :].reshape(1, -1), class_mav, metric_type=metric_type)
# 
#         weibull_model[label]['mean_vec'] = class_mav
#         weibull_model[label]['distances'] = av_distance
# 
#         mr = libmr.MR()
# 
#         tail_size_fix = min(tail_size, av_distance.shape[1])
#         tails_to_fit = sorted(av_distance[0, :])[-tail_size_fix:]
#         mr.fit_high(tails_to_fit, tail_size_fix)
# 
#         weibull_model[label]['weibull_model'] = mr
# 
#     return weibull_model


def main(test_dataloader):

    model_save_path = ''

    img_all = []
    label_all = []
    AV_all = []
    score_all = []
    ridlist = []
    viscodelist = []

    net = DSA_3D_CNN().to(device)
    net.load_state_dict(
        torch.load("/home/lxs/DSA-3D-CNN/DSA-3D-CNN/modelsave/DSA_3D_CNN_best.pth"))  # ///////////
    net.eval()
    with torch.no_grad():
        for step, (img, label, _, rid, viscode) in enumerate(test_dataloader):
            img = img.float().to(device)
            # print("label.type1= ", type(label))
            label = label.long().to(device)
            # print("label.type2= ", type(label))


            
            rid = rid.long().to(device)
            viscode = np.array(viscode, dtype=str)
            
            # print("viscode.type= ", type(viscode))
            # print("viscode= ", viscode)
            
            out = net(img)
            softmaxout = copy.deepcopy(out)
            out = out.cpu().detach()
            out = np.array(out)
            # print("type(out)= ", type(out))
            # print("out.shape= ", out.shape)
            # print("out= ", out)
            # net_2 = nn.Softmax(dim=1)
            # out = net_2(out)
            label = label.cpu().detach()
            label = np.array(label)

            rid = rid.cpu().detach()
            rid = np.array(rid)
            
            # print("rid.type= ", type(rid))
            # print("rid= ", rid)
            # viscode = viscode.cpu().detach()
            # viscode = np.array(viscode)

            # print("type(label)= ", type(label))
            # print("label= ", label)

            # test_y_true, test_y_pred = UT.assemble_labels(step, test_y_true, test_y_pred, label, out)


            net_2 = nn.Softmax(dim=1)
            softmaxout = net_2(softmaxout)
            softmaxout = softmaxout.cpu().detach()
            softmaxout = np.array(softmaxout)
            for labelline in label:
                label_all.append(labelline)
            for outline in out:
                AV_all.append(outline)
            for softmaxoutline in softmaxout:
                score_all.append(softmaxoutline)

            for ridline in rid:
                ridlist.append(ridline)
            for viscodeline in viscode:
                viscodelist.append(viscodeline)

            # print("type(softmaxout)= ", type(softmaxout))
            # print("softmaxout.shape= ", softmaxout.shape)
            # print("softmaxout= ", softmaxout)

    # AV_all = np.array(AV_all)
    # print("type(AV_all)= ", type(AV_all))
    # print("np.shape(AV_all)= ", np.shape(AV_all))
    # score_all = np.array(score_all)
    # print("type(score_all)= ", type(score_all))
    # print("np.shape(score_all)= ", np.shape(score_all))

    ridlist = pd.DataFrame(ridlist)
    ridlist.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/openmax_data/DSA_3D_CNN_smc_test_ridlist.csv', index=0)
    viscodelist = pd.DataFrame(viscodelist)
    viscodelist.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/openmax_data/DSA_3D_CNN_smc_test_viscodelist.csv', index=0)

    AV_all = pd.DataFrame(AV_all)
    AV_all.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/openmax_data/DSA_3D_CNN_smc_test_out.csv', index=0)
    score_all = pd.DataFrame(score_all)
    score_all.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/openmax_data/DSA_3D_CNN_smc_test_softmax.csv', index=0)
    label_all = pd.DataFrame(label_all)
    label_all.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/openmax_data/DSA_3D_CNN_smc_test_label.csv', index=0)



    # print('Start train weibull model ..... ..... ..... ..... ..... ..... ..... ')
    # weibull_model = weibull_fit(model_save_path + '.pkl', AV_all.reshape(-1, 2),
    #                             score_all.reshape(-1, 2), label_all)
    # print('Complete train layer4_weibull model ..... ..... ..... ..... ..... ..... ..... ')
    #
    # score = score_all.reshape(-1, 2)
    # result_y4 = []
    # for i in range(score.shape[0]):
    #     openmax, softmax = recalibrate_scores_2v(score[i, :], weibull_model, weibull_model.keys(), alpha_rank=2)
    #     # openmax, softmax = recalibrate_scores(score[i, :], y3_weibull_model, y3_weibull_model.keys(), alpha_rank=2)
    #     # print(openmax, result_tr[11][i], result_tr[12][i])
    #     result_y4.append(openmax)
    #
    # openmax_result = get_divided(result_y4, len(result_y4), 3)


if __name__ == '__main__':
    GPU = 0
    BATCH_SIZE = 8
    EPOCHS = 1

    LR = 0.000015
    LOSS_WEIGHTS = torch.tensor([1., 1.])

    device = torch.device('cuda:' + str(GPU) if torch.cuda.is_available() else 'cpu')

    train_hist = []
    val_hist = []
    test_performance = []
    test_y_true = np.asarray([])
    test_y_pred = np.asarray([])
    full_path = np.asarray([])

    filepath = '/home/lxs/ncomms2022-main/MRI_process/test_data/processed/npy/'
    trainlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/train.csv'
    validlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/valid.csv'
    testlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/test.csv'
    
    # trainlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/train.csv'
    # validlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/valid.csv'
    # testlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/test.csv'

    test_dataset = Dataset_Early_Fusion(filepath=filepath, label_file=testlabel_file)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False,
                                                      drop_last=True)
    print("----------------- start--------------")
    main(test_dataloader)
    print("----------------- finish!!!--------------")










































    # GPU = 0
    # BATCH_SIZE = 8
    # EPOCHS = 2
    #
    # LR = 0.000015
    # LOSS_WEIGHTS = torch.tensor([1., 1.])
    #
    # device = torch.device('cuda:' + str(GPU) if torch.cuda.is_available() else 'cpu')
    #
    # train_hist = []
    # val_hist = []
    # test_performance = []
    # test_y_true = np.asarray([])
    # test_y_pred = np.asarray([])
    # full_path = np.asarray([])
    #
    # filepath = '/home/lxs/ncomms2022-main/MRI_process/test_data/processed/npy/'
    # trainlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/train.csv'
    # validlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/valid.csv'
    # testlabel_file = '/home/lxs/DSA-3D-CNN/DSA-3D-CNN/csv/test.csv'
    #
    #
    # print("-----------------test start--------------")
    # test_dataset = Dataset_Early_Fusion(filepath=filepath, label_file=testlabel_file)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False,
    #                                                   drop_last=False)
    # test(test_dataloader)
    #
    # print("------------------test end---------------")

