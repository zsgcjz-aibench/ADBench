#-*- coding: utf-8 -*-
import numpy as np
import skimage.transform as skTrans
import os
import pickle
from path import Path
from argparse import ArgumentParser
from torch.utils.data import Dataset
import csv
#from slicing_test import random_slicing
import random
import pandas as pd
import cv2
from torch.utils.data import DataLoader
import nibabel as nib
current_dir = Path(__file__).parent.abspath()

def random_slicing(dataset_num, num_clients):
    """Slice a dataset randomly and equally for IID.

    Args：
        dataset (torch.utils.data.Dataset): a dataset for slicing.
        num_clients (int):  the number of client.

    Returns：
        dict: ``{ 0: indices of dataset, 1: indices of dataset, ..., k: indices of dataset }``
    """
    num_items = int(dataset_num / num_clients)
    print("num_items:", num_items)
    dict_users, all_idxs = {}, [i for i in range(dataset_num)]
    for i in range(num_clients):
        dict_users[i] = list(
            np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users

def get_AD_risk(raw):
    x1, x2 = raw[0, :, :, :], raw[1, :, :, :]
    risk = np.exp(x2) / (np.exp(x1) + np.exp(x2))
    return risk

def read_csv_complete(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames, labels, demors, rids, visits = [], [], [], [], []
    for line in your_list:
        try:
            age = float(line[0])
            mmse = float(line[3])
            gender = [0, 1] if line[5] == 'Female' else [1, 0]
            demor = [(age-70.0)/10.0] + gender + [(mmse-27)/2]
        except:
            continue

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
        demors.append(demor)

    return filenames, labels, demors, rids, visits

class MLP_Data(Dataset):
    def __init__(self, Data_dir, filename,indexs, choice, seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        self.roi_threshold = 0.6
        self.roi_count = 200

        self.Data_list, self.Label_list = get_idx_data(filename, indexs)

        if choice == 'count':
            self.select_roi_count()
        else:
            self.select_roi_thres()

        self.risk_list = [get_AD_risk(np.load(Data_dir + filename + '.npy'))[self.roi] for filename in self.Data_list]

        self.in_size = self.risk_list[0].shape[0]

    def select_roi_thres(self):
        self.roi = np.load('/valid_MCC.npy')
        self.roi = self.roi > self.roi_threshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i % 3 != 0 or j % 2 != 0 or k % 3 != 0:
                        self.roi[i, j, k] = False

    def select_roi_count(self):
        self.roi = np.load('/valid_MCC.npy')
        tmp = []
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i % 3 != 0 or j % 2 != 0 or k % 3 != 0: continue
                    tmp.append((self.roi[i, j, k], i, j, k))
        tmp.sort()
        tmp = tmp[-self.roi_count:]
        self.roi = self.roi != self.roi
        for _, i, j, k in tmp:
            self.roi[i, j, k] = True

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        risk = self.risk_list[idx]

        return risk, label

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.Label_list)), float(self.Label_list.count(0)), float(
            self.Label_list.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.Label_list]
        return weights, count0 / count1

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    filenames = [a[8] for a in your_list[1:]]
    labels = [0 if a[2] == 'CN' else 1 for a in your_list[1:]]

    return filenames, labels

def get_idx_data(filename, idx):
    data_list, targets = read_csv(filename)

    subdataset = [data_list[i] for i in idx]
    sublabelset = [targets[i] for i in idx]

    return subdataset, sublabelset

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

class Dataset_model(Dataset):
    def __init__(self, filepath,label_file, indexs):
        self.Data_list, self.Label_list = get_idx_data(label_file, indexs)
        self.filepaths = filepath

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):

        label = self.Label_list[idx]

        full_path = self.Data_list[idx]

        im = np.load(self.filepaths + full_path + '.npy')

        im = np.reshape(im, (1, 182, 218, 182))

        return im, label

class Dataset_model_Dynamic(Dataset):
    def __init__(self, filepath,
                 label_file, indexs):
        self.Data_list, self.Label_list = get_idx_data(label_file, indexs)
        self.filepaths = filepath

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]

        full_path = self.Data_list[idx]

        im = np.load(self.filepaths + full_path + '.npy')

        im = skTrans.resize(im, (110, 110, 110), order=1, preserve_range=True)
        im = np.reshape(im, (110, 110, 110, 1))
        im = get_dynamic_image(im)
        im = np.expand_dims(im, 0)
        im = np.concatenate([im, im, im], 0)
        im = np.array(im, dtype=np.float32)
        return im, label


def preprocess(args):

    print("current_dir:", current_dir)

    train_data_file = '/train_federal.csv'
    valid_data_files = '/valid_federal.csv'

    test_data_files = '/site_testset.csv'

    # #
    train_data = pd.read_csv(train_data_file)
    train_num = len(train_data)

    train_idxs = random_slicing(train_num, args.client_num_in_total)

    valid_data = pd.read_csv(valid_data_files)
    valid_num = len(valid_data)

    valid_idxs = random_slicing(valid_num, args.client_num_in_total)


    test_data = pd.read_csv(test_data_files)
    test_num = len(test_data)


    test_idxs = random_slicing(test_num, args.client_num_in_total)

    all_trainsets = []
    all_validsets = []

    Data_dir = '/DPMs/fcn_exp/'

    for train_indices, valid_indices, test_indices in zip(train_idxs.values(), valid_idxs.values(), test_idxs.values()):
        all_trainsets.append(MLP_Data(Data_dir, train_data_file, train_indices, 'count'))
        all_validsets.append(MLP_Data(Data_dir,valid_data_files,valid_indices, 'count'))

    for i in range(args.client_num_in_total):
        with open("{}/pickles_random_client_5/client_{}.pkl".format(current_dir, i), "wb") as file:
            pickle.dump((all_trainsets[i], all_validsets[i]), file)



def preprocess_Dynamic(args):


    print("current_dir:", current_dir)

    train_data_file = '/train_federal.csv'
    valid_data_files = '/valid_federal.csv'

    test_data_files = '/site_testset.csv'

    # #
    train_data = pd.read_csv(train_data_file)
    train_num = len(train_data)

    train_idxs = random_slicing(train_num, args.client_num_in_total)

    valid_data = pd.read_csv(valid_data_files)
    valid_num = len(valid_data)
    valid_idxs = random_slicing(valid_num, args.client_num_in_total)

    test_data = pd.read_csv(test_data_files)
    test_num = len(test_data)

    test_idxs = random_slicing(test_num, args.client_num_in_total)

    all_trainsets = []
    all_validsets = []

    all_testsets = []

    Data_dir = '/ADNI/npy/'

    for train_indices, valid_indices, test_indices in zip(train_idxs.values(), valid_idxs.values(), test_idxs.values()):
        all_trainsets.append(Dataset_model_Dynamic(Data_dir, train_data_file, train_indices))  # 遍历字典里的一个键里面全部的值，就是一个客户端样本的索引
        all_validsets.append(Dataset_model_Dynamic(Data_dir, valid_data_files, valid_indices))
        all_testsets.append(Dataset_model_Dynamic(Data_dir, test_data_files, test_indices))

    for i in range(args.client_num_in_total):
        with open("{}/pickles_Dynamic/client_{}.pkl".format(current_dir, i), "wb") as file:
            pickle.dump((all_trainsets[i], all_validsets[i], all_testsets[i]), file)


if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=5)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    preprocess(args)

    print("######  preprocess_Dynamic  #########")

    preprocess_Dynamic(args)
















