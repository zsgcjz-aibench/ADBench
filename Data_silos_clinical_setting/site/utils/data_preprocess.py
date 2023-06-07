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

class Dataset_model(Dataset):
    def __init__(self, image_path, data_site, label_site):
        # self.files = UT.read_csv(label_file)
        # self.Data_list, self.Label_list = read_csv(label_file)
        self.Data_list, self.Label_list = data_site, label_site
        # print("self.files:", self.Data_list)
        self.filepaths = image_path

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):

        label = self.Label_list[idx]

        # label = 0 if label == 'CN' else 1

        full_path = self.Data_list[idx]

        im = np.load(self.filepaths + full_path + '.npy')
        #im = nib.load(self.filepaths + full_path + '.nii').get_fdata()  # (110, 110, 110)
        #print("im.shape:", im.shape)  # (110, 110, 110)
        #im = np.asarray(im).astype(np.float32)
        # im = skTrans.resize(im, (110, 110, 110), order=1, preserve_range=True)
        # im = np.reshape(im, (1, 110, 110, 110))
        im = np.reshape(im, (1, 182, 218, 182))  # (1, 110, 110, 110) # 182, 218, 182

        return im, label  # output image shape [T,C,W,H]

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

class Dataset_model_Dynamic(Dataset):
    def __init__(self, image_path, data_site, label_site):
        self.Data_list, self.Label_list = data_site, label_site
        # print("self.files:", self.Data_list)
        self.filepaths = image_path

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
def get_AD_risk(raw):
    x1, x2 = raw[0, :, :, :], raw[1, :, :, :]
    risk = np.exp(x2) / (np.exp(x1) + np.exp(x2))
    return risk

def read_csv_complete(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    # print("your_list:", your_list)
    filenames, labels, demors, rids, visits = [], [], [], [], []
    for line in your_list:
        # print("line:", line)
        try:
            # print("###############")
            age = float(line[0])  # list(map(float, line[2:5]))
            mmse = float(line[3])
            gender = [0, 1] if line[5] == 'Female' else [1, 0]
            # print("gender:", gender)
            # print("mmse:", mmse)
            # print("age:", age)
            demor = [(age-70.0)/10.0] + gender + [(mmse-27)/2]   # [(age-70.0)/10.0] +
            # print("demor:", demor)
            # demor = [demor[0]] + gender + demor[2:]
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
        # print("filenames:", filenames)
        # print("###:", len(rids))
    return filenames, labels, demors, rids, visits


class MLP_Data(Dataset):
    def __init__(self, Data_dir, imagenames, labels, choice, seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        self.roi_threshold = 0.6
        self.roi_count = 200
        if choice == 'count':
            self.select_roi_count()
        else:
            self.select_roi_thres()
        # print("file:", imagenames)

        # self.path = '{}{}.csv'.format(file, stage)
        # self.path = file
        # print("self.path:", self.path)
        self.Label_list = labels
        self.Data_list = imagenames

        # self.Data_list, self.Label_list, self.demor_list, self.rid_list, self.visit_list = read_csv_complete(self.path)


        self.risk_list = [get_AD_risk(np.load(Data_dir + filename + '.npy'))[self.roi] for filename in self.Data_list]

        self.in_size = self.risk_list[0].shape[0]

    def select_roi_thres(self):
        self.roi = np.load('/home/lxj/mxx/dataset_no_rid_repeat/FCN_MLP/FCN_model/DPMs/fcn_exp0/valid_data1_MCC.npy')
        self.roi = self.roi > self.roi_threshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i % 3 != 0 or j % 2 != 0 or k % 3 != 0:
                        self.roi[i, j, k] = False

    def select_roi_count(self):
        self.roi = np.load('/home/lxj/mxx/dataset_no_rid_repeat/FCN_MLP/FCN_model/DPMs/fcn_exp0/valid_data1_MCC.npy')
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


def load_data():

    all_data = pd.read_csv('/site_data.csv')

    print("len(all_data):", len(all_data))

    site_counts = all_data['SITE'].value_counts()

    site_counts_index = site_counts.index  # 站点名称

    client = 20
    train_data_file = []
    train_data_label = []
    for i in range(0, client):
        temp_site = int(site_counts_index[i])
        temp_data = all_data[all_data['SITE'].isin([temp_site])]
        filenames = []
        labels = []
        for index, row in temp_data.iterrows():
            filenames.append(row['SavePath'])
            if row['DX'] == 'CN':
                label = 0
            else:
                label = 1
            labels.append(label)

        train_data_file.append(filenames)
        train_data_label.append(labels)

    valid_data_file = []
    valid_data_label = []

    for i in range(client, client + client):
        temp_site = int(site_counts_index[i])
        temp_data = all_data[all_data['SITE'].isin([temp_site])]
        filenames = []
        labels = []
        for index, row in temp_data.iterrows():
            filenames.append(row['SavePath'])
            if row['DX'] == 'CN':
                label = 0
            else:
                label = 1
            labels.append(label)

        valid_data_file.append(filenames)
        valid_data_label.append(labels)


    test_data = pd.read_csv('/site_testset.csv')

    site_5 = test_data['SITE'].value_counts()
    site_counts_index_5 = site_5.index
    test_data_file_grobal = []
    test_data_label_grobal = []

    data_grabal = pd.DataFrame()
    for i in range(0, 5):
        temp_site = int(site_counts_index_5[i])
        temp_data = test_data[test_data['SITE'].isin([temp_site])]
        data_grabal = data_grabal.append(temp_data)

    for index, row in data_grabal.iterrows():
        test_data_file_grobal.append(row['SavePath'])
        if row['DX'] == 'CN':
            label = 0
        else:
            label = 1
        test_data_label_grobal.append(label)
    print("test_data_label_grobal_len", len(test_data_label_grobal))

    return train_data_file, train_data_label, valid_data_file, valid_data_label, test_data_file_grobal, test_data_label_grobal


def preprocess(args):

    print("current_dir:", current_dir)
    Data_dir = '/ADNI/npy/'

    train_data_file, train_data_label, valid_data_file, valid_data_label, test_data_file_grobal, test_data_label_grobal = load_data()

    trainsets_site = []

    for data, label in zip(train_data_file, train_data_label):
        trainsets_site.append(Dataset_model(Data_dir, data, label))

    validsets_site = []
    for data, label in zip(valid_data_file, valid_data_label):
        validsets_site.append(Dataset_model(Data_dir, data, label))

    testsets_site = Dataset_model(Data_dir, test_data_file_grobal, test_data_label_grobal)

    for i in range(args.client_num_in_total):
        with open("{}/pickles_client/client_site_{}.pkl".format(current_dir, i), "wb") as file:
            pickle.dump((trainsets_site[i], validsets_site[i]), file)

    with open("{}/pickles_client/client_site_global.pkl".format(current_dir), "wb") as file:
        pickle.dump((testsets_site), file)


def preprocess_Dynamic(args):


    Data_dir = '/ADNI/npy/'
    train_data_file, train_data_label, valid_data_file, valid_data_label, test_data_file_grobal, test_data_label_grobal = load_data()

    trainsets_site = []

    for data, label in zip(train_data_file, train_data_label):
        trainsets_site.append(Dataset_model_Dynamic(Data_dir, data, label))

    validsets_site = []
    for data, label in zip(valid_data_file, valid_data_label):
        validsets_site.append(Dataset_model_Dynamic(Data_dir, data, label))

    testsets_site = Dataset_model_Dynamic(Data_dir, test_data_file_grobal, test_data_label_grobal)

    for i in range(args.client_num_in_total):
        with open("{}/pickles_site_client/client_site_{}.pkl".format(current_dir, i), "wb") as file:
            pickle.dump((trainsets_site[i], validsets_site[i]), file)

    with open("{}/pickles_site_Dynamic_client/client_site_global.pkl".format(current_dir), "wb") as file:
        pickle.dump((testsets_site), file)


if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=20)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()


    preprocess(args)

    print("######  preprocess_Dynamic  #########")

    preprocess_Dynamic(args)






