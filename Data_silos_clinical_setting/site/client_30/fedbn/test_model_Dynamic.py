import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

from NeurIPS.Data_silos_clinical_setting.site.nets.models import DSA_3D_CNN, Dynamic_images_VGG, LRP, Vox, _MLP_A
from NeurIPS.Data_silos_clinical_setting.site.utils.data_preprocess import Dataset_model, Dataset_model_Dynamic,MLP_Data
from NeurIPS.Data_silos_clinical_setting.site.utils.eval_index_cal import  assemble_labels, cal_CI_plot_close
def get_global(files, batch_size):
    with open(files, "rb") as file:
        grobal_set = pickle.load(file)
    grobal_loader = DataLoader(grobal_set, batch_size, shuffle=True, drop_last=True)
    return grobal_loader


def test(model, test_loader, loss_fun):
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

        output = model(data)

        test_loss += loss_fun(output, target).item()

        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
        test_y_true, test_y_pred = assemble_labels(step, test_y_true, test_y_pred, target, output)

    preds = torch.max(test_y_pred, 1)[1]

    preds = np.array(preds)
    cal_CI_plot_close(preds, test_y_pred, test_y_true)
    return test_loss / len(test_loader), correct / len(test_loader.dataset)


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    batch_size = 8
    print("size:", batch_size)
    test_loader = get_global('/client_global.pkl',  batch_size) # Need to adjust the path

    server_model = Dynamic_images_VGG()
    loss_fun = nn.CrossEntropyLoss()

    print("server_model:", server_model)

    checkpoint = torch.load('/fed_domainnet/Dynamic/fedbn') # Need to adjust the path

    print("a_iter:", checkpoint['a_iter'])
    print("best_acc:", checkpoint['best_acc'])

    server_model.load_state_dict(checkpoint['server_model'])

    with torch.no_grad():
        test_loss, test_acc = test(server_model, test_loader, loss_fun)
        print(' {:<11s}| Test  Acc: {:.4f}'.format('server_model', test_acc))
