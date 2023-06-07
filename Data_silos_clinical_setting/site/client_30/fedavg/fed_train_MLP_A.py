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



def get_adni(client_num, batch_size):
    trainloaders = []
    validloaders = []
    for i in range(client_num):
        with open("/client_site_{}.pkl".format(i), "rb") as file:
            trainset, validset, testset = pickle.load(file)

            trainloader = DataLoader(trainset, batch_size, shuffle=True, drop_last=True)
            validloader = DataLoader(validset, batch_size, shuffle=True, drop_last=True)

            trainloaders.append(trainloader)

            validloaders.append(validloader)

    return trainloaders, validloaders


def get_global(files, batch_size):
    with open(files, "rb") as file:
        grobal_set = pickle.load(file)

    grobal_loader = DataLoader(grobal_set, batch_size, shuffle=True, drop_last=True)

    return grobal_loader


def train(model, train_loader, optimizer, loss_fun, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)
        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / len(train_iter), correct / num_data


def train_prox(args, model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output = model(data)

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / len(test_loader), correct / len(test_loader.dataset)


################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=300, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='fedavg', help='[FedBN | FedAvg | FedProx]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='MLP_A', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
    args = parser.parse_args()

    exp_folder = 'fed_domainnet'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))

    log = args.log

    # if log:
    log_path = os.path.join('MLP_A', exp_folder)
    print("log_path:", log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logfile = open(os.path.join(log_path, '{}.log'.format(args.mode)), 'a')
    print("logfile:", logfile)

    logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logfile.write('===Setting===\n')
    logfile.write('    lr: {}\n'.format(args.lr))
    logfile.write('    batch: {}\n'.format(args.batch))
    logfile.write('    iters: {}\n'.format(args.iters))

    datasets = ['client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8', 'client9',
                'client10', 'client11', 'client12', 'client13', 'client14', 'client15', 'client16', 'client17',
                'client18', 'client19', 'client20', 'client21', 'client22', 'client23', 'client24', 'client25',
                'client26', 'client27', 'client28', 'client29',
                'client30']
    client_num = len(datasets)
    print("client_num:", client_num)
    batch_size = 8
    print("batch_size:", batch_size)
    train_loaders, val_loaders = get_adni(client_num, batch_size)

    test_loader = get_global('client_global.pkl', batch_size)

    # setup model
    server_model = _MLP_A(in_size=200, fil_num=100, drop_rate=0.5).to(device)
    loss_fun = nn.CrossEntropyLoss()

    print("device:", device)
    print("server_model:", server_model)

    client_weights = [1 / client_num for i in range(client_num)]

    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    best_changed = False

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
        start_iter = int(checkpoint['a_iter']) + 1

        print('Resume training from epoch {}'.format(start_iter))
    else:
        # log the best for each model on all datasets
        best_epoch = 0
        best_acc = [0. for j in range(client_num)]
        start_iter = 0

    start = datetime.now()
    # Start training
    for a_iter in range(start_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx, model in enumerate(models):
                if args.mode.lower() == 'fedprox':
                    # skip the first server model(random initialized)
                    if a_iter > 0:
                        train_loss, train_acc = train_prox(args, model, train_loaders[client_idx],
                                                           optimizers[client_idx], loss_fun, device)
                    else:
                        train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx],
                                                      loss_fun, device)
                else:
                    train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun,
                                                  device)

        with torch.no_grad():
            # Aggregation
            server_model, models = communication(args, server_model, models, client_weights)

            # stop1 = datetime.now()
            # Report loss after aggregation
            for client_idx, model in enumerate(models):
                print("client_idx:", client_idx)
                train_loss, train_acc = test(model, train_loaders[client_idx], loss_fun, device)
                print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss,
                                                                                     train_acc))
                if args.log:
                    logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx],
                                                                                                   train_loss,
                                                                                                   train_acc))

            # Validation
            val_acc_list = [None for j in range(client_num)]
            for client_idx, model in enumerate(models):
                val_loss, val_acc = test(model, val_loaders[client_idx], loss_fun, device)
                val_acc_list[client_idx] = val_acc
                print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss,
                                                                                   val_acc))
                if args.log:
                    logfile.write(
                        ' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss,
                                                                                       val_acc))

            # Record best
            if np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    best_acc[client_idx] = val_acc_list[client_idx]
                    best_epoch = a_iter
                    best_changed = True
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(datasets[client_idx], best_epoch,
                                                                                  best_acc[client_idx]))
                    if args.log:
                        logfile.write(
                            ' Best site-{:<10s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], best_epoch,
                                                                                       best_acc[client_idx]))

            if best_changed:
                print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                logfile.write(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))

                torch.save({
                    'server_model': server_model.state_dict(),
                    'best_epoch': best_epoch,
                    'best_acc': best_acc,
                    'a_iter': a_iter
                }, SAVE_PATH)
                best_changed = False

                with torch.no_grad():
                    test_loss, test_acc = test(server_model, test_loader, loss_fun, device)
                    print(' {:<11s}| Test  Acc: {:.4f}'.format('server_model', test_acc))

        end = datetime.now()
        print("#####  Aggregation " + str(a_iter))
        print("Running time: ", end - start)

        if log:
            logfile.flush()
    if log:
        logfile.flush()
        logfile.close()
