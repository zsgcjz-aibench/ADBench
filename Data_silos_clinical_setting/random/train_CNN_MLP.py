import os
import sys
import pdb
import json
import torch
import numpy as np
import pandas as pd
from time import time as get_timestamp
from math import ceil, floor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from config import root_path, mri_path
from models import _CNN_Bone, MLP
from tqdm import trange, tqdm
import torch.nn.functional as F
import time
from sklearn import metrics
import matplotlib.pyplot as plt

class MRIModel(object):

    def __init__(self, dataset_path):
        self.train_set = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
        self.valid_set = pd.read_csv(os.path.join(dataset_path, 'valid.csv'))
        self.test_set = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        with open(os.path.join(root_path, 'task_config.json'), 'r') as file:
            self.config = json.loads(file.read())
            file.close()

        self.backbone = _CNN_Bone(self.config['backbone']).to(self.device)
        self.mlp = MLP(self.backbone.size, self.config['COG']).to(self.device)

        self.loss = CrossEntropyLoss().to(self.device)
        self.backbone_optim = Adam(self.backbone.parameters(), lr=self.config['backbone']['lr'], betas=(0.5, 0.999))
        self.mlp_optim = Adam(self.mlp.parameters(), lr=self.config['COG']['lr'])
        
        self.metric_optim = 0
        self.best_epoch = 0

        self.batch_size = 32
        self.train_accuracy = []
        self.train_auc = []
        self.train_loss = []
        self.valid_accuracy = []
        self.valid_auc = []
        self.valid_loss = []

    def timeit(method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if 'log_time' in kw:
                name = kw.get('log_name', method.__name__.upper())
                kw['log_time'][name] = int((te - ts) * 1000)
            else:
                print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
            return result

        return timed


    def load_dataset(self, stage, shuffle=True):
        # 选择要载入的数据集
        if stage not in ['train', 'valid', 'test']:
            raise ValueError('stage: "train", "valid" ,"test"!')
        dataset = getattr(self, '{}_set'.format(stage))

        if shuffle:
            index = np.arange(dataset.shape[0], dtype='int')
            np.random.shuffle(index)
            dataset.index = range(dataset.shape[0])
            dataset = dataset.loc[index, :]
            dataset.index = range(dataset.shape[0])


        filenames = dataset['filename'].values
        labels = dataset[['status']].values
        rids = dataset[['RID']].values
        viscodes = dataset[['VISCODE']].values
        num_sample = len(filenames)
        num_batch = floor(num_sample / self.batch_size)

        for batch in range(num_batch):
            start = batch * self.batch_size
            if batch == num_batch - 1:
                end = num_sample
            else:
                end = start + self.batch_size

            batch_inputs = np.zeros((end - start, 1, 182, 218, 182), 'float32')
            batch_labels = labels[start:end]
            batch_rids = rids[start:end]
            batch_viscode = viscodes[start:end]
            for arr_i, fn_i in enumerate(range(start, end)):
                fp = os.path.join(mri_path, filenames[fn_i] + '.npy')
                mri = np.expand_dims(np.load(fp).astype('float32'), axis=0)
                batch_inputs[arr_i, :] = mri
            yield batch_inputs, batch_labels, batch_rids, batch_viscode


    def train(self, num_epochs, save_path=None):
        t = trange(num_epochs, desc=' ', leave=True)
        # for epoch in range(num_epochs):
        for e in t:
            start_time = get_timestamp()

            self.train_an_epoch()

            train_accuracy, train_loss, train_auc, valid_accuracy, valid_loss, valid_auc = self.valid_model(int(e) + 1)
            self.train_accuracy.append(train_accuracy)
            self.train_auc.append(train_auc)
            self.train_loss.append(train_loss.cpu())
            self.valid_accuracy.append(valid_accuracy)
            self.valid_auc.append(valid_auc)
            self.valid_loss.append(valid_loss.cpu())
            print('Epoch {}'.format(int(e) + 1))
            print('train_accuracy = {}'.format(train_accuracy))
            print('train_auc = {}'.format(train_auc))
            print('train_loss = {}'.format(train_loss))
            print('valid_accuracy = {}'.format(valid_accuracy))
            print('valid_auc = {}'.format(valid_auc))
            print('valid_loss = {}'.format(valid_loss))
            print('times: {:.2f}s'.format(get_timestamp() - start_time))
            print()

            self.adjust_learning_rate(int(e), num_epochs)
            # 保存模型
            if save_path is not None:
                self.save_model(save_path, int(e) + 1)
            #寻找最优的epoch
            if valid_auc > self.metric_optim:
                self.metric_optim = valid_auc
                self.best_epoch = int(e) + 1


    def train_an_epoch(self):

        self.backbone.train(True)
        self.mlp.train(True)


        self.backbone.zero_grad()
        self.mlp.zero_grad()
        for batch_inputs, batch_labels, batch_rids, batch_viscode in self.load_dataset('train'):
            batch_inputs = torch.tensor(batch_inputs).to(self.device)
            batch_labels = torch.tensor(batch_labels).to(self.device)
            batch_rids = torch.tensor(batch_rids).to(self.device)
            # batch_viscode = torch.tensor(batch_viscode).to(self.device)

            batch_labels = F.one_hot(batch_labels.to(torch.int64), num_classes=2).float().squeeze()

            batch_preds, batch_av = self.mlp(self.backbone(batch_inputs))
            # print("batch_preds= ", batch_preds)
            # print("batch_labels= ", batch_labels)

            batch_loss = self.loss(batch_preds, batch_labels)
            batch_loss.backward()
        self.backbone_optim.step()
        self.mlp_optim.step()

    def valid_model(self, epoch):
        # 置为非训练状态
        self.backbone.train(False)
        self.mlp.train(False)

        # 计算accuracy和loss
        with torch.no_grad():
            train_accuracy, train_loss, train_auc = self.compute_accuracy_and_loss('train', epoch)
            valid_accuracy, valid_loss, valid_auc = self.compute_accuracy_and_loss('valid', epoch)
        return train_accuracy, train_loss, train_auc, valid_accuracy, valid_loss, valid_auc


    def compute_accuracy_and_loss(self, stage, epoch):
        true_count = 0
        loss_sum = 0
        loss_count = 0

        val_preds = []
        val_label = []
        val_rids = []
        val_viscodes = []
        val_avs = []
        for batch_inputs, batch_labels, batch_rids, batch_viscode in self.load_dataset(stage):
            batch_inputs = torch.tensor(batch_inputs).to(self.device)
            batch_labels = torch.tensor(batch_labels).to(self.device)
            batch_rids = torch.tensor(batch_rids).to(self.device)
            # batch_viscode = torch.tensor(batch_viscode).to(self.device)

            batch_preds, batch_av = self.mlp(self.backbone(batch_inputs))

            batch_labels = F.one_hot(batch_labels.to(torch.int64), num_classes=2).float().squeeze()

            loss_sum = loss_sum + self.loss(batch_preds, batch_labels)
            loss_count = loss_count + 1

            pred_labels = batch_preds.data.cpu().squeeze().numpy().argmax(axis=1)
            real_labels = batch_labels.data.cpu().squeeze().numpy().argmax(axis=1)
            true_count = true_count + sum(pred_labels == real_labels)

            batch_rids = batch_rids.cpu().numpy()
            batch_viscode = batch_viscode
            pred_labels = batch_preds.data.cpu().squeeze().numpy()
            real_labels = batch_labels.cpu().numpy()
            batch_av = batch_av.data.cpu().squeeze().numpy()
            for i in range(batch_rids.shape[0]):
                val_rids.append(batch_rids[i][0])
                val_viscodes.append(batch_viscode[i][0])
                val_preds.append(pred_labels[i, :])
                val_avs.append(batch_av[i, :])
                val_label.append(real_labels[i, 1])



        accuracy = true_count / getattr(self, '{}_set'.format(stage)).shape[0]
        auc = metrics.roc_auc_score(batch_labels.data.cpu().squeeze().numpy()[:,1], batch_preds.data.cpu().squeeze().numpy()[:, 1])
        loss = loss_sum / loss_count


        val_preds = np.array(val_preds)
        val_avs = np.array(val_avs)
        save_result = []

        #for index in range(0, len(val_rids)):
        #    save_result.append([val_rids[index], val_viscodes[index], val_preds[index, 0], val_preds[index, 1],
        #                        val_avs[index, 0], val_avs[index, 1],
        #                        val_label[index]])

        #save_result = pd.DataFrame(save_result)
        #save_path = '/home/lxj/mxx/dataset_no_rid_repeat/ncomms2022-main_xie/binary_category_mri/valid_pred/' + str(stage) + str(epoch) + '_y_pred.csv'

        #save_result.to_csv(save_path, index=0, header=['rid', 'viscode', 's1', 's2', 'o1', 'o2', 'label'])

        return accuracy, loss, auc

    def save_model(self, save_path, index=None):
        dir_path = os.path.dirname(save_path)
        if index is None:
            exist_index_list = [int(fn.rsplit('.', 1)[0].rsplit('_', 1)[-1]) for fn in os.listdir(dir_path)]
            index = max(exist_index_list) + 1 if exist_index_list else 1
        torch.save(self.backbone.state_dict(), os.path.join(dir_path, 'backbone_{}.pth'.format(index)))
        torch.save(self.mlp.state_dict(), os.path.join(dir_path, 'mlp_{}.pth'.format(index)))

    def adjust_learning_rate(self, epoch, num_epochs):
        if epoch not in [round(num_epochs * 0.3333), round(num_epochs * 0.6666)]:
            return None
        for param_group in self.backbone_optim.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
        for param_group in self.mlp_optim.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2

    def draw_eval_plot(self, x, show=True, save_path=None):
        text_fontdict = {'size': 20}
        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        plt.title('accuracy')
        plt.plot(x, self.train_accuracy, '-o', label='train_acc')
        plt.plot(x, self.valid_accuracy, '-o', label='valid_acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        i = np.array(self.train_accuracy).argmax()
        best_train_acc = (x[i], self.train_accuracy[i])
        i = np.array(self.valid_accuracy).argmax()
        best_valid_acc = (x[i], self.valid_accuracy[i])
        plt.text(*best_train_acc, s='({},{:.4f})'.format(*best_train_acc), fontdict=text_fontdict)
        plt.text(*best_valid_acc, s='({},{:.4f})'.format(*best_valid_acc), fontdict=text_fontdict)

        plt.subplot(1, 2, 2)
        plt.title('loss')
        plt.plot(x, self.train_loss, '-o', label='train_loss')
        plt.plot(x, self.valid_loss, '-o', label='valid_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        i = np.array(self.train_loss).argmin()
        best_train_loss = (x[i], self.train_loss[i])
        i = np.array(self.valid_loss).argmin()
        best_valid_loss = (x[i], self.valid_loss[i])
        plt.text(*best_train_loss, s='({},{:.4f})'.format(*best_train_loss), fontdict=text_fontdict)
        plt.text(*best_valid_loss, s='({},{:.4f})'.format(*best_valid_loss), fontdict=text_fontdict)

        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()


    def loadWeights(self, best_epoch):
        self.load_backbone_Weights(best_epoch)
        self.load_MLP_weights(best_epoch)

    def load_backbone_Weights(self, best_epoch):
        # print("self.checkpoint_dir= ", self.checkpoint_dir)
        best_epoch = best_epoch
        target_file = './checkpoint_dir/' + 'backbone_' + str(best_epoch) + '.pth'
        # print('loading ', target_file)
        weights = torch.load(target_file, map_location=lambda storage, loc: storage) #storage.cuda(self.device)
        try:
            self.backbone.load_state_dict(weights)
        except:
            self.backbone.load_state_dict(remove_module(weights))

    def load_MLP_weights(self, best_epoch):
        target_file = './checkpoint_dir/' + 'mlp_' + str(best_epoch) + '.pth'

        try:
            self.mlp.load_state_dict(torch.load(target_file))
        except:
            self.mlp.load_state_dict(remove_module(torch.load(target_file, map_location=torch.device('cpu'))))


    def test(self):
        # 置为非训练状态
        self.backbone.train(False)
        self.mlp.train(False)
        print("self.best_epoch= ", self.best_epoch)
        self.loadWeights(self.best_epoch)
        # 计算accuracy和loss
        with torch.no_grad():
            test_accuracy, test_loss, test_auc = self.compute_accuracy_and_loss('test', epoch)
        return test_accuracy, test_loss, test_auc
            



if __name__ == '__main__':

    epoch = 100
    mri_model = MRIModel('../data/')
    mri_model.train(epoch, './checkpoint_dir/')
    # mri_model.save_model()
    mri_model.draw_eval_plot(range(1, epoch+1), show=False, save_path='/training_eval.png')
    test_accuracy, test_loss, test_auc = mri_model.test()
    print("test_accuracy={}, test_loss={}, test_auc={} ".format(test_accuracy, test_loss, test_auc))

    
