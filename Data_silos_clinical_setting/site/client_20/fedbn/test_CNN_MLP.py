from __future__ import print_function, division
import json
from config import root_path, mri_path

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datetime import datetime
import torch
from torch import nn, optim
import time
import copy
import argparse
import numpy as np
import pickle

from torch.utils.data import DataLoader
import random
from NeurIPS.Data_silos_clinical_setting.site.nets.models import DSA_3D_CNN, Dynamic_images_VGG, LRP, Vox, _CNN_Bone, MLP
from NeurIPS.Data_silos_clinical_setting.site.utils.data_preprocess import Dataset_model, Dataset_model_Dynamic,MLP_Data
from NeurIPS.Data_silos_clinical_setting.site.utils.eval_index_cal import  assemble_labels, cal_CI_plot_close
def get_global(files, batch_size):
    with open(files, "rb") as file:
        grobal_set = pickle.load(file)

    grobal_loader = DataLoader(grobal_set, batch_size, shuffle=True, drop_last=True)

    return grobal_loader


def test(model, backbone, test_loader, loss_fun):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    test_y_true = []
    test_y_pred = []

    for step, (data, target) in enumerate(test_loader):
        data = data.float()
        target = target.long()
        targets.append(target.detach().cpu().numpy())
        output, openmax = model(backbone(data))

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
        test_y_true, test_y_pred = assemble_labels(step, test_y_true, test_y_pred, target, output)

    preds = torch.max(test_y_pred, 1)[1]
    preds = np.array(preds)

    cal_CI_plot_close(preds, test_y_pred, test_y_true)

    return test_loss / len(test_loader), correct / len(test_loader.dataset)


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    batch_size = 8
    print("batch_size:", batch_size)
    test_loader = get_global('/client_global.pkl', batch_size)  # Need to adjust the path


    with open(os.path.join(root_path, 'task_config.json'), 'r') as file:
        config = json.loads(file.read())
        file.close()

    server_model1 = _CNN_Bone(config['backbone'])
    server_model = MLP(server_model1.size, config['COG'])

    loss_fun = nn.CrossEntropyLoss()

    checkpoint = torch.load('/fed_domainnet/CNN_MLP/fedbn')  # Need to adjust the path

    server_model.load_state_dict(checkpoint['server_model'])

    print("device:", device)
    print("server_model:", server_model)
    print("a_iter:", checkpoint['a_iter'])
    print("best_acc:", checkpoint['best_acc'])

    with torch.no_grad():
        test_loss, test_acc = test(server_model, server_model1, test_loader, loss_fun)
        print(' {:<11s}| Test  Acc: {:.4f}'.format('server_model', test_acc))




















