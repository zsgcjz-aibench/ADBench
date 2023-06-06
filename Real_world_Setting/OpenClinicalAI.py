import tensorflow as tf

from tensorflow.keras import layers

from tensorflow import keras

from tensorflow.keras.initializers import Constant

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input

from tensorflow.keras import Model

from tensorflow.keras.layers import Masking

import numpy as np

import datetime

import os
import random
import math

import copy

import matplotlib.pyplot as plt

import HierarchicalOpenNet

from sklearn.metrics.pairwise import paired_distances
import libmr
import pickle

import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from collections import Counter

from sklearn.metrics import roc_curve, auc






os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(1)

tf.random.set_seed(50)


class CustomMultiLossLayer(layers.Layer):
    def __init__(self, nb_outputs=2, model_type='acm', loss_type='cross_entropy', name="CustomMultiLossLayer",
                 methond_list=[52788, 48116, 29096, 32765, 42595, 14594, 3652, 10867, 2558, 1937, 1947, 960], **kwargs):
        self.nb_outputs = nb_outputs
        self.model_type = model_type
        self.loss_type = loss_type
        self.methond_list = methond_list
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def get_config(self):
        config = {"nb_outputs": self.nb_outputs, "model_type": self.model_type, "loss_type": self.loss_type,
                  "methond_list": self.methond_list}
        base_config = super(CustomMultiLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        index = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            # loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
            if self.loss_type == 'cross_entropy':
                # a = (100703 + 125768 + 203369) / (2 * 100703)
                # b = (100703 + 125768 + 203369) / (2 * (125768 + 203369))
                loss += K.sum(
                    precision * (-(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[0],
                    -1)
                # loss += K.sum(precision * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred) + log_var[0], -1)
            elif self.loss_type == 'M_BCE':
                # a = (100703 + 125768 + 203369) / (2 * 100703)
                # b = (100703 + 125768 + 203369) / (2 * (125768 + 203369))
                # return tf.reduce_mean(-(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred)))
                ad = 100703
                cn = 125768
                mci = 203369

                total_acm = ad + cn + mci

                smci = 78527
                pmci = 51614

                total_sp = smci + pmci

                # x2 = 52112
                # x3 = 64616
                # x4 = 61292
                # x5 = 87406
                # x6 = 76820
                # x7 = 35056
                # x8 = 10568
                # x9 = 37199
                # x10 = 9740
                # x11 = 4605
                # x12 = 8070
                # x13 = 4452

                x2 = self.methond_list[0]
                x3 = self.methond_list[1]
                x4 = self.methond_list[2]
                x5 = self.methond_list[3]
                x6 = self.methond_list[4]
                x7 = self.methond_list[5]
                x8 = self.methond_list[6]
                x9 = self.methond_list[7]
                x10 = self.methond_list[8]
                x11 = self.methond_list[9]
                x12 = self.methond_list[10]
                x13 = self.methond_list[11]

                # x_total = 100703+125768+203369
                x_total = 100703 + 125768

                # 0 52788 48116 29096 32765 42595 14594  3652 10867  2558  1937  1947 960
                ac_x2 = self.methond_list[0]
                ac_x3 = self.methond_list[1]
                ac_x4 = self.methond_list[2]
                ac_x5 = self.methond_list[3]
                ac_x6 = self.methond_list[4]
                ac_x7 = self.methond_list[5]
                ac_x8 = self.methond_list[6]
                ac_x9 = self.methond_list[7]
                ac_x10 = self.methond_list[8]
                ac_x11 = self.methond_list[9]
                ac_x12 = self.methond_list[10]
                ac_x13 = self.methond_list[11]

                ac_x_total = 100703 + 125768

                if self.model_type == 'acm':
                    if index == 0:
                        a = total_acm / (2 * ad)
                        b = total_acm / (2 * (cn + mci))
                        loss += K.sum(precision * (
                        -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[0],
                                      -1)
                    elif index == 1:
                        a = total_acm / (2 * cn)
                        b = total_acm / (2 * (ad + mci))
                        loss += K.sum(precision * (
                        -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[0],
                                      -1)
                    elif index == 2:
                        a = total_acm / (2 * mci)
                        b = total_acm / (2 * (ad + cn))
                        loss += K.sum(precision * (
                        -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[0],
                                      -1)
                if self.model_type == 'sp':
                    if index == 0:
                        a = total_sp / (2 * smci)
                        b = total_sp / (2 * pmci)
                        loss += K.sum(precision * (
                        -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[0],
                                      -1)
                    elif index == 1:
                        a = total_sp / (2 * pmci)
                        b = total_sp / (2 * smci)
                        loss += K.sum(precision * (
                        -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[0],
                                      -1)
                if self.model_type == 'methond':

                    if index == 0:
                        a = x_total / (2 * x2)
                        b = x_total / (2 * (x_total - x2))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 1:
                        a = x_total / (2 * x3)
                        b = x_total / (2 * (x_total - x3))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 2:
                        a = x_total / (2 * x4)
                        b = x_total / (2 * (x_total - x4))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 3:
                        a = x_total / (2 * x5)
                        b = x_total / (2 * (x_total - x5))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 4:
                        a = x_total / (2 * x6)
                        b = x_total / (2 * (x_total - x6))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 5:
                        a = x_total / (2 * x7)
                        b = x_total / (2 * (x_total - x7))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 6:
                        a = x_total / (2 * x8)
                        b = x_total / (2 * (x_total - x8))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 7:
                        a = x_total / (2 * x9)
                        b = x_total / (2 * (x_total - x9))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 8:
                        a = x_total / (2 * x10)
                        b = x_total / (2 * (x_total - x10))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 9:
                        a = x_total / (2 * x11)
                        b = x_total / (2 * (x_total - x11))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 10:
                        a = x_total / (2 * x12)
                        b = x_total / (2 * (x_total - x12))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 11:
                        a = x_total / (2 * x13)
                        b = x_total / (2 * (x_total - x13))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                if self.model_type == 'ac_methond':

                    if index == 0:
                        a = ac_x_total / (2 * ac_x2)
                        b = ac_x_total / (2 * (ac_x_total - ac_x2))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 1:
                        a = ac_x_total / (2 * ac_x3)
                        b = ac_x_total / (2 * (ac_x_total - ac_x3))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 2:
                        a = ac_x_total / (2 * ac_x4)
                        b = ac_x_total / (2 * (ac_x_total - ac_x4))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 3:
                        a = ac_x_total / (2 * ac_x5)
                        b = ac_x_total / (2 * (ac_x_total - ac_x5))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 4:
                        a = ac_x_total / (2 * ac_x6)
                        b = ac_x_total / (2 * (ac_x_total - ac_x6))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 5:
                        a = ac_x_total / (2 * ac_x7)
                        b = ac_x_total / (2 * (ac_x_total - ac_x7))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 6:
                        a = ac_x_total / (2 * ac_x8)
                        b = ac_x_total / (2 * (ac_x_total - ac_x8))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 7:
                        a = ac_x_total / (2 * ac_x9)
                        b = ac_x_total / (2 * (ac_x_total - ac_x9))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 8:
                        a = ac_x_total / (2 * ac_x10)
                        b = ac_x_total / (2 * (ac_x_total - ac_x10))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 9:
                        a = ac_x_total / (2 * ac_x11)
                        b = ac_x_total / (2 * (ac_x_total - ac_x11))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 10:
                        a = ac_x_total / (2 * ac_x12)
                        b = ac_x_total / (2 * (ac_x_total - ac_x12))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)
                    elif index == 11:
                        a = ac_x_total / (2 * ac_x13)
                        b = ac_x_total / (2 * (ac_x_total - ac_x13))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) + log_var[
                                          0],
                                      -1)

            elif self.loss_type == 'MSE':
                loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
            index += 1
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        # return ys_true,ys_pred
        return ys_true, ys_pred
        # K.concatenate(inputs, -1)

layer1_num = 96
layer2_num = 96
layer3_num = 64
dropout_rate = 0.4
steps = 127

statistics_transform = np.zeros(13, dtype=int)
acm_methond_dic_train = {}
acm_methond_dic_eval = {}
g_evidence_dic = {}
trans_methond_count = [0]

# region  #骨架网络
mask_layer = Masking(mask_value=-4.0, input_shape=(None, 2090), name='Bone_Mask')
Lstm1 = layers.Bidirectional(layers.LSTM(layer1_num, return_sequences=True, dropout=dropout_rate, name='Bone_LSTM1'),
                             name='Bone_BiLSTM1')
nor1 = layers.BatchNormalization(name='Bone_Nor1')
Lstm2 = layers.Bidirectional(layers.LSTM(layer2_num, return_sequences=True, dropout=dropout_rate, name='Bone_LSTM2'),
                             name='Bone_BiLSTM2')
nor2 = layers.BatchNormalization(name='Bone_Nor2')
Lstm3 = layers.Bidirectional(layers.LSTM(layer3_num, dropout=dropout_rate, name='Bone_LSTM3'), name='Bone_BiLSTM3')
nor3 = layers.BatchNormalization(name='Bone_Nor3')
# endregion

# region  # 2-13类检查手段 分类网络
# 1位基础信息，第2位认知信息，第3位认知检查，第4位精神信息，
# 第5位行为功能检查，第6位体格检查,第7位血液检查，第8位尿液检查，
# 第9位MRI检查，第10位18F-FDG PET, 第11位18F-AV45 PET，
# 第12位基因检测，第13位脑脊液检测。
dense_menthod1 = layers.Dense(32, activation='relu', name='menthod_Dense1')
drop_menthod1 = layers.Dropout(dropout_rate, name='menthod_Drop1')

# region method 2
dense_menthod_2_1 = layers.Dense(16, activation='relu', name='menthod_2_1_Dense1')
drop_menthod_2_1 = layers.Dropout(0.4, name='menthod_2_1_Drop1')
dense_menthod_2_2 = layers.Dense(16, activation='relu', name='menthod_2_2_Dense2')
dense_menthod_2_3 = layers.Dense(1, activation='sigmoid', name='menthod_2_3_Dense3')
# endregion

# region  method 3
dense_menthod_3_1 = layers.Dense(16, activation='relu', name='menthod_3_1_Dense1')
drop_menthod_3_1 = layers.Dropout(0.4, name='menthod_3_1_Drop1')
dense_menthod_3_2 = layers.Dense(16, activation='relu', name='menthod_3_2_Dense2')
dense_menthod_3_3 = layers.Dense(1, activation='sigmoid', name='menthod_3_3_Dense3')
# endregion

# region  method 4
dense_menthod_4_1 = layers.Dense(16, activation='relu', name='menthod_4_1_Dense1')
drop_menthod_4_1 = layers.Dropout(0.4, name='menthod_4_1_Drop1')
dense_menthod_4_2 = layers.Dense(16, activation='relu', name='menthod_4_2_Dense2')
dense_menthod_4_3 = layers.Dense(1, activation='sigmoid', name='menthod_4_3_Dense3')
# endregion

# region  method 5
dense_menthod_5_1 = layers.Dense(16, activation='relu', name='menthod_5_1_Dense1')
drop_menthod_5_1 = layers.Dropout(0.4, name='menthod_5_1_Drop1')
dense_menthod_5_2 = layers.Dense(16, activation='relu', name='menthod_5_2_Dense2')
dense_menthod_5_3 = layers.Dense(1, activation='sigmoid', name='menthod_5_3_Dense3')
# endregion

# region  method 6
dense_menthod_6_1 = layers.Dense(16, activation='relu', name='menthod_6_1_Dense1')
drop_menthod_6_1 = layers.Dropout(0.4, name='menthod_6_1_Drop1')
dense_menthod_6_2 = layers.Dense(16, activation='relu', name='menthod_6_2_Dense2')
dense_menthod_6_3 = layers.Dense(1, activation='sigmoid', name='menthod_6_3_Dense3')
# endregion

# region  method 7
dense_menthod_7_1 = layers.Dense(16, activation='relu', name='menthod_7_1_Dense1')
drop_menthod_7_1 = layers.Dropout(0.4, name='menthod_7_1_Drop1')
dense_menthod_7_2 = layers.Dense(16, activation='relu', name='menthod_7_2_Dense2')
dense_menthod_7_3 = layers.Dense(1, activation='sigmoid', name='menthod_7_3_Dense3')
# endregion

# region  method 8
dense_menthod_8_1 = layers.Dense(16, activation='relu', name='menthod_8_1_Dense1')
drop_menthod_8_1 = layers.Dropout(0.4, name='menthod_8_1_Drop1')
dense_menthod_8_2 = layers.Dense(16, activation='relu', name='menthod_8_2_Dense2')
dense_menthod_8_3 = layers.Dense(1, activation='sigmoid', name='menthod_8_3_Dense3')
# endregion

# region  method 9
dense_menthod_9_1 = layers.Dense(16, activation='relu', name='menthod_9_1_Dense1')
drop_menthod_9_1 = layers.Dropout(0.4, name='menthod_9_1_Drop1')
dense_menthod_9_2 = layers.Dense(16, activation='relu', name='menthod_9_2_Dense2')
dense_menthod_9_3 = layers.Dense(1, activation='sigmoid', name='menthod_9_3_Dense3')
# endregion

# region  method 10
dense_menthod_10_1 = layers.Dense(16, activation='relu', name='menthod_10_1_Dense1')
drop_menthod_10_1 = layers.Dropout(0.4, name='menthod_10_1_Drop1')
dense_menthod_10_2 = layers.Dense(16, activation='relu', name='menthod_10_2_Dense2')
dense_menthod_10_3 = layers.Dense(1, activation='sigmoid', name='menthod_10_3_Dense3')
# endregion

# region  method 11
dense_menthod_11_1 = layers.Dense(16, activation='relu', name='menthod_11_1_Dense1')
drop_menthod_11_1 = layers.Dropout(0.4, name='menthod_11_1_Drop1')
dense_menthod_11_2 = layers.Dense(16, activation='relu', name='menthod_11_2_Dense2')
dense_menthod_11_3 = layers.Dense(1, activation='sigmoid', name='menthod_11_3_Dense3')
# endregion

# region  method 12
dense_menthod_12_1 = layers.Dense(16, activation='relu', name='menthod_12_1_Dense1')
drop_menthod_12_1 = layers.Dropout(0.4, name='menthod_12_1_Drop1')
dense_menthod_12_2 = layers.Dense(16, activation='relu', name='menthod_21_2_Dense2')
dense_menthod_12_3 = layers.Dense(1, activation='sigmoid', name='menthod_12_3_Dense3')
# endregion

# region  method 13
dense_menthod_13_1 = layers.Dense(16, activation='relu', name='menthod_13_1_Dense1')
drop_menthod_13_1 = layers.Dropout(0.4, name='menthod_13_1_Drop1')
dense_menthod_13_2 = layers.Dense(16, activation='relu', name='menthod_13_2_Dense2')
dense_menthod_13_3 = layers.Dense(1, activation='sigmoid', name='menthod_13_3_Dense3')


# endregion

# endregion


# 检查手段分类网络
def get_methond_model_4v():
    inp_methond = Input(shape=(None, 2090), name='inp_methond')
    x = Lstm1(inp_methond)
    x = nor1(x)
    x = Lstm2(x)
    x = nor2(x)
    x = Lstm3(x)
    x = nor3(x)

    # menthod_x = tf.concat([x, ad_x, cn_x, mci_x, s_x, p_x], axis=1)
    menthod_x = dense_menthod1(x)
    menthod_x = drop_menthod1(menthod_x)

    inp_ad = Input(shape=(1), name='inp_ad')
    inp_cn = Input(shape=(1), name='inp_cn')
    inp_mci = Input(shape=(1), name='inp_mci')

    menthod_x = tf.concat([menthod_x, inp_ad, inp_cn, inp_mci], axis=1)

    # region method 2
    # m2_x = dense_menthod_2_1(menthod_x)
    # m2_x = drop_menthod_2_1(m2_x)
    m2_x = dense_menthod_2_2(menthod_x)
    m2_x = dense_menthod_2_3(m2_x)
    # endregion

    # region  method 3
    # m3_x = dense_menthod_3_1(menthod_x)
    # m3_x = drop_menthod_3_1(m3_x)
    m3_x = dense_menthod_3_2(menthod_x)
    m3_x = dense_menthod_3_3(m3_x)
    # endregion

    # region  method 4
    # m4_x = dense_menthod_4_1(menthod_x)
    # m4_x = drop_menthod_4_1(m4_x)
    m4_x = dense_menthod_4_2(menthod_x)
    m4_x = dense_menthod_4_3(m4_x)
    # endregion

    # region  method 5
    # m5_x = dense_menthod_5_1(menthod_x)
    # m5_x = drop_menthod_5_1(m5_x)
    m5_x = dense_menthod_5_2(menthod_x)
    m5_x = dense_menthod_5_3(m5_x)
    # endregion

    # region  method 6
    # m6_x = dense_menthod_6_1(menthod_x)
    # m6_x = drop_menthod_6_1(m6_x)
    m6_x = dense_menthod_6_2(menthod_x)
    m6_x = dense_menthod_6_3(m6_x)
    # endregion

    # region  method 7
    # m7_x = dense_menthod_7_1(menthod_x)
    # m7_x = drop_menthod_7_1(m7_x)
    m7_x = dense_menthod_7_2(menthod_x)
    m7_x = dense_menthod_7_3(m7_x)
    # endregion

    # region  method 8
    # m8_x = dense_menthod_8_1(menthod_x)
    # m8_x = drop_menthod_8_1(m8_x)
    m8_x = dense_menthod_8_2(menthod_x)
    m8_x = dense_menthod_8_3(m8_x)
    # endregion

    # region  method 9
    # m9_x = dense_menthod_9_1(menthod_x)
    # m9_x = drop_menthod_9_1(m9_x)
    m9_x = dense_menthod_9_2(menthod_x)
    m9_x = dense_menthod_9_3(m9_x)
    # endregion

    # region  method 10
    # m10_x = dense_menthod_10_1(menthod_x)
    # m10_x = drop_menthod_10_1(m10_x)
    m10_x = dense_menthod_10_2(menthod_x)
    m10_x = dense_menthod_10_3(m10_x)
    # endregion

    # region  method 11
    # m11_x = dense_menthod_11_1(menthod_x)
    # m11_x = drop_menthod_11_1(m11_x)
    m11_x = dense_menthod_11_2(menthod_x)
    m11_x = dense_menthod_11_3(m11_x)
    # endregion

    # region  method 12
    # m12_x = dense_menthod_12_1(menthod_x)
    # m12_x = drop_menthod_12_1(m12_x)
    m12_x = dense_menthod_12_2(menthod_x)
    m12_x = dense_menthod_12_3(m12_x)
    # endregion

    # region  method 13
    # m13_x = dense_menthod_13_1(menthod_x)
    # m13_x = drop_menthod_13_1(m13_x)
    m13_x = dense_menthod_13_2(menthod_x)
    m13_x = dense_menthod_13_3(m13_x)
    # endregion

    return Model([inp_methond, inp_ad, inp_cn, inp_mci],
                 [m2_x, m3_x, m4_x, m5_x, m6_x, m7_x, m8_x, m9_x, m10_x, m11_x, m12_x, m13_x])


def get_trainable_model_methond(prediction_model, loss_type='cross_entropy', methond_list=[52788, 48116, 29096, 32765, 42595, 14594, 3652, 10867, 2558, 1937, 1947,960]):
    inp = Input(shape=(None, 2090), name='inp_methond')

    inp_ad = Input(shape=(1), name='inp_ad')
    inp_cn = Input(shape=(1), name='inp_cn')
    inp_mci = Input(shape=(1), name='inp_mci')

    y2_pred, y3_pred, y4_pred, y5_pred, y6_pred, y7_pred, y8_pred, y9_pred, y10_pred, y11_pred, y12_pred, y13_pred = prediction_model(
        [inp, inp_ad, inp_cn, inp_mci])
    y2_true = Input(shape=(1), name='y2_true_methond')
    y3_true = Input(shape=(1), name='y3_true_methond')
    y4_true = Input(shape=(1), name='y4_true_methond')
    y5_true = Input(shape=(1), name='y5_true_methond')
    y6_true = Input(shape=(1), name='y6_true_methond')
    y7_true = Input(shape=(1), name='y7_true_methond')
    y8_true = Input(shape=(1), name='y8_true_methond')
    y9_true = Input(shape=(1), name='y9_true_methond')
    y10_true = Input(shape=(1), name='y10_true_methond')
    y11_true = Input(shape=(1), name='y11_true_methond')
    y12_true = Input(shape=(1), name='y12_true_methond')
    y13_true = Input(shape=(1), name='y13_true_methond')
    out = CustomMultiLossLayer(nb_outputs=12, model_type='ac_methond', loss_type=loss_type, methond_list=methond_list)(
        [y2_true, y3_true, y4_true, y5_true, y6_true, y7_true, y8_true, y9_true, y10_true, y11_true, y12_true, y13_true,
         y2_pred, y3_pred, y4_pred, y5_pred, y6_pred, y7_pred, y8_pred, y9_pred, y10_pred, y11_pred, y12_pred,
         y13_pred])
    return Model(
        [inp, inp_ad, inp_cn, inp_mci, y2_true, y3_true, y4_true, y5_true, y6_true, y7_true, y8_true, y9_true, y10_true,
         y11_true, y12_true, y13_true], out)

acm_learning_rate = 0.0005
sp_learning_rate = 0.0005
methond_learning_rate = 0.0005

acm_opt_rms = keras.optimizers.RMSprop(lr=acm_learning_rate, rho=0.9, epsilon=None)
sp_opt_rms = keras.optimizers.RMSprop(lr=sp_learning_rate, rho=0.9, epsilon=None)
methond_opt_rms = keras.optimizers.RMSprop(lr=methond_learning_rate, rho=0.9, epsilon=None)

acm_opt_adam = keras.optimizers.Adam(lr=acm_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
sp_opt_adam = keras.optimizers.Adam(lr=sp_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
methond_opt_adam = keras.optimizers.Adam(lr=methond_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         amsgrad=False)

acm_opt_sdg = keras.optimizers.SGD(lr=acm_learning_rate, momentum=0.9, nesterov=False)
sp_opt_sdg = keras.optimizers.SGD(lr=sp_learning_rate, momentum=0.9, nesterov=False)
methond_opt_sdg = keras.optimizers.SGD(lr=methond_learning_rate, momentum=0.9, nesterov=False)


numEpochs = 10000

acm_batch_size = 64
acm_batch_size_ev = 512
acm_batch_size_te = 256

sp_batch_size = 64
sp_batch_size_ev = 1024
sp_batch_size_te = 256

methond_batch_size = 32
methond_batch_size_ev = 1024
methond_batch_size_te = 256

model_save_path = '/data/huangyunyou/ODMLCS_MODEL/'

smc_test_set = '/data/huangyunyou/TFRecord_8v/smc_test.tfrecord'

ac_training_set = '/data/huangyunyou/TFRecord_2v/ac_train.tfrecord'
ac_validation_set = '/data/huangyunyou/TFRecord_2v/ac_eval.tfrecord'
ac_test_set = '/data/huangyunyou/TFRecord_2v/ac_test.tfrecord'

mci_test_set = '/data/huangyunyou/TFRecord_2v/mci_test.tfrecord'


# methond_training_set='/data/huangyunyou/TFRecord/'
# methond_validation_set='/data/huangyunyou/TFRecord/'
# methond_test_set='/data/huangyunyou/TFRecord/'


def acm_read_and_decode(example_proto):
    features = {"data": tf.io.VarLenFeature(tf.string),
                "label_ad": tf.io.FixedLenFeature([], tf.string),
                "label_cn": tf.io.FixedLenFeature([], tf.string),
                "label_mci": tf.io.FixedLenFeature([], tf.string),
                "rid": tf.io.FixedLenFeature([], tf.string),
                "viscode": tf.io.FixedLenFeature([], tf.string)}

    parsed_features = tf.io.parse_single_example(example_proto, features)

    data = parsed_features["data"]
    # data = tf.io.decode_raw(data, tf.float32)
    data = tf.sparse.to_dense(data)  # 使用VarLenFeature读入的是一个sparse_tensor，用该函数进行转换
    data = tf.io.decode_raw(data, tf.float32)

    ad = parsed_features["label_ad"]
    ad = tf.io.decode_raw(ad, tf.float32)

    cn = parsed_features["label_cn"]
    cn = tf.io.decode_raw(cn, tf.float32)

    mci = parsed_features["label_mci"]
    mci = tf.io.decode_raw(mci, tf.float32)

    rid = parsed_features["rid"]
    # rid = tf.io.decode_raw(rid, tf.string)

    viscode = parsed_features["viscode"]
    # viscode = tf.io.decode_raw(viscode, tf.string)

    return data, ad, cn, mci, rid, viscode


def acm_read_and_decode_eva(example_proto):
    features = {"data": tf.io.VarLenFeature(tf.string),
                "label_ad": tf.io.FixedLenFeature([], tf.string),
                "label_cn": tf.io.FixedLenFeature([], tf.string),
                "label_mci": tf.io.FixedLenFeature([], tf.string),
                "rid": tf.io.FixedLenFeature([], tf.string),
                "viscode": tf.io.FixedLenFeature([], tf.string)}

    parsed_features = tf.io.parse_single_example(example_proto, features)

    data = parsed_features["data"]
    # data = tf.io.decode_raw(data, tf.float32)
    data = tf.sparse.to_dense(data)  # 使用VarLenFeature读入的是一个sparse_tensor，用该函数进行转换
    data = tf.io.decode_raw(data, tf.float32)
    data = tf.reshape(data, [-1, 2090])

    ad = parsed_features["label_ad"]
    ad = tf.io.decode_raw(ad, tf.float32)

    cn = parsed_features["label_cn"]
    cn = tf.io.decode_raw(cn, tf.float32)

    mci = parsed_features["label_mci"]
    mci = tf.io.decode_raw(mci, tf.float32)

    rid = parsed_features["rid"]
    # rid = tf.io.decode_raw(rid, tf.string)

    viscode = parsed_features["viscode"]
    # viscode = tf.io.decode_raw(viscode, tf.string)

    return {'inp_acm': data, 'acm_ad_Dense3': ad, 'acm_cn_Dense3': cn, 'acm_mci_Dense3': mci}


def sp_read_and_decode(example_proto):
    features = {"data": tf.io.VarLenFeature(tf.string),
                "label_s": tf.io.FixedLenFeature([], tf.string),
                "label_p": tf.io.FixedLenFeature([], tf.string),
                "rid": tf.io.FixedLenFeature([], tf.string),
                "viscode": tf.io.FixedLenFeature([], tf.string)}

    parsed_features = tf.io.parse_single_example(example_proto, features)

    data = parsed_features["data"]
    # data = tf.io.decode_raw(data, tf.float32)
    data = tf.sparse.to_dense(data)  # 使用VarLenFeature读入的是一个sparse_tensor，用该函数进行转换
    data = tf.io.decode_raw(data, tf.float32)

    s = parsed_features["label_s"]
    s = tf.io.decode_raw(s, tf.float32)

    p = parsed_features["label_p"]
    p = tf.io.decode_raw(p, tf.float32)

    rid = parsed_features["rid"]
    # rid = tf.io.decode_raw(rid, tf.string)

    viscode = parsed_features["viscode"]
    # viscode = tf.io.decode_raw(viscode, tf.string)

    return data, s, p, rid, viscode


def get_data_set(path, map_methond, batch_size, data_type=0):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(map_methond)
    dataset = dataset.repeat(numEpochs)
    dataset = dataset.shuffle(buffer_size=20000)
    # dataset = dataset.batch(batch_size)
    if data_type == 0:
        dataset = dataset.padded_batch(batch_size, padded_shapes=([1, steps * 2090], [1, ], [1, ], [1, ], [], []),
                                       padding_values=(-4.0, -4.0, -4.0, -4.0, '-4.0', '-4.0'))
    elif data_type == 1:
        dataset = dataset.padded_batch(batch_size, padded_shapes=([1, steps * 2090], [1, ], [1, ], [], []),
                                       padding_values=(-4.0, -4.0, -4.0, '-4.0', '-4.0'))
    elif data_type == 2:
        dataset = dataset.padded_batch(batch_size, padded_shapes=(
        [1, steps * 2090], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [], []),
                                       padding_values=(
                                       -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0,
                                       '-4.0', '-4.0'))
    return dataset


def get_data_test_set(path, map_methond, batch_size, data_type=0):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(map_methond)
    # dataset = dataset.batch(batch_size)
    if data_type == 0:
        dataset = dataset.padded_batch(batch_size, padded_shapes=([1, steps * 2090], [1, ], [1, ], [1, ], [], []),
                                       padding_values=(-4.0, -4.0, -4.0, -4.0, '-4.0', '-4.0'))
    elif data_type == 1:
        dataset = dataset.padded_batch(batch_size, padded_shapes=([1, steps * 2090], [1, ], [1, ], [], []),
                                       padding_values=(-4.0, -4.0, -4.0, '-4.0', '-4.0'))
    elif data_type == 2:
        dataset = dataset.padded_batch(batch_size, padded_shapes=(
            [1, steps * 2090], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [],
            []),
                                       padding_values=(
                                           -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0,
                                           '-4.0', '-4.0'))
    return dataset


def early_stop(histy, monitor='loss', min_delta=0, patience=0):
    l = len(histy)
    if l < 1:
        return False
    elif l < patience:
        return False
    elif l >= patience:
        tmp_histy = histy[l - patience:l]
        value = tmp_histy[0][monitor]
        s_flage = True
        for i in range(1, patience):
            tmp_value = tmp_histy[i][monitor]
            if (value - tmp_value) >= min_delta:
                s_flage = False
                break
        return s_flage


def if_model_save(histy, monitor='loss', patience=0):
    l = len(histy)
    if l < 1:
        return 5000000, 5000000, 5000000
    elif l < patience:
        return 5000000, 5000000, 5000000
    elif l >= patience:
        tmp_histy = histy[l - patience:l]

        sum = 0
        for i in range(0, patience):
            tmp_value = tmp_histy[i][monitor]
            sum += tmp_value
        means = sum / patience

        var = 0
        for i in range(0, patience):
            tmp_value = tmp_histy[i][monitor]
            var += (tmp_value - means) ** 2
        means_var = var / patience

        return means, means_var, histy[l - 1][monitor]


def get_data_set_eav(path, map_methond, batch_size, data_type=0):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(map_methond)
    # dataset = dataset.batch(batch_size)
    if data_type == 0:
        # {'inp_acm':data},{'acm_ad_Dense3':ad,'acm_cn_Dense3':cn,'acm_mci_Dense3':mci}
        dataset = dataset.padded_batch(batch_size, padded_shapes={'inp_acm': [steps, 2090], 'acm_ad_Dense3': [1, ],
                                                                  'acm_cn_Dense3': [1, ], 'acm_mci_Dense3': [1, ]},
                                       padding_values={'inp_acm': -4.0, 'acm_ad_Dense3': -4.0, 'acm_cn_Dense3': -4.0,
                                                       'acm_mci_Dense3': -4.0})
    elif data_type == 1:
        dataset = dataset.padded_batch(batch_size, padded_shapes=([steps, 2090], [1, ], [1, ]),
                                       padding_values=(-4.0, -4.0, -4.0))
    elif data_type == 2:
        dataset = dataset.padded_batch(batch_size, padded_shapes=(
        [steps, 2090], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ], [1, ]),
                                       padding_values=(
                                       -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0))
    return dataset


ac_tr = get_data_set(ac_training_set, sp_read_and_decode, acm_batch_size, 1)
ac_tr_it = ac_tr.__iter__()
ac_va = get_data_set(ac_validation_set, sp_read_and_decode, acm_batch_size_ev, 1)
ac_va_it = ac_va.__iter__()
ac_te = get_data_test_set(ac_test_set, sp_read_and_decode, acm_batch_size_te, 1)
ac_te_it = ac_te.__iter__()

mci_te = get_data_test_set(mci_test_set, sp_read_and_decode, acm_batch_size_te, 1)
mci_te_it = mci_te.__iter__()

smc_te = get_data_test_set(smc_test_set, acm_read_and_decode, acm_batch_size_te)
smc_te_it = smc_te.__iter__()


# methond_tr=get_data_set(methond_training_set,methond_read_and_decode,methond_batch_size)
# methond_tr_it = methond_tr.__iter__()
# methond_va=get_data_set(methond_validation_set,methond_read_and_decode,methond_batch_size_ev)
# methond_va_it = methond_va.__iter__()
# methond_te=get_data_test_set(methond_test_set,methond_read_and_decode,methond_batch_size_te)
# methond_te_it = methond_te.__iter__()
methond_tr = ''
methond_tr_it = ''
methond_va = ''
methond_va_it = ''
methond_te = ''
methond_te_it = ''

metrics_auc = tf.keras.metrics.AUC()
metrics_accuracy = tf.keras.metrics.BinaryAccuracy()


def get_one_hot(data_y, batch_len):
    # print(data_y)
    # print(batch_len)
    ret_value = []
    class_num = len(data_y)
    for i in range(0, batch_len):
        tmp_one_hot = np.zeros(class_num)
        for j in range(class_num):
            if data_y[j][i][0] == 1.0:
                tmp_one_hot[j] = 1
        ret_value.append(tmp_one_hot)
    # print(ret_value)
    return tf.convert_to_tensor(np.array(ret_value))


def get_one_hot_2v(data_y, batch_len):
    # print(data_y)
    # print(batch_len)
    ret_value = []
    class_num = len(data_y)
    for i in range(0, batch_len):
        tmp_one_hot = np.zeros(class_num)
        for j in range(class_num):
            tmp_one_hot[j] = data_y[j][i]
        ret_value.append(tmp_one_hot)
    # print(ret_value)
    return ret_value


def get_divided(data_y, batch_len, class_num):
    ret_value = []
    # print(data_y)
    for i in range(class_num):
        ret_value.append([])

    for i in range(0, batch_len):
        tmp_one_hot = data_y[i]
        for j in range(class_num):
            ret_value[j].append(tmp_one_hot[j])
    # print(ret_value)
    return ret_value


def get_singel_label(data_y, batch_len, class_num):
    ret_value = []
    for i in range(batch_len):
        for j in range(class_num):
            if data_y[j][i] == 1.0:
                ret_value.append(j)
                break
    return np.array(ret_value)


def get_auc(ys_true, ys_pred, isReturnAccuracy=False, openmax_rate=[]):
    nb_outputs = len(ys_true)
    ys_pred_open_max = []

    if len(openmax_rate) > 0:
        for i in range(nb_outputs):
            ys_pred_open_max.append([])

    for j in range(len(openmax_rate)):
        tmp_open_max = openmax_rate[j]
        # print('##################')
        # print(tmp_open_max)

        index = np.argmax(tmp_open_max)
        # print(index)

        thr_low = False
        for l in range(nb_outputs):
            if index == (nb_outputs - 1):
                if l == index:
                    ys_pred_open_max[l].append(1.0)
                else:
                    ys_pred_open_max[l].append(0.0)
            else:
                if l == index:
                    if tmp_open_max[l] >= 0.9:
                        ys_pred_open_max[l].append(1.0)
                        thr_low = True
                    else:
                        ys_pred_open_max[l].append(0.0)
                else:
                    if l == (nb_outputs - 1):
                        if thr_low:
                            ys_pred_open_max[l].append(0.0)
                        else:
                            ys_pred_open_max[l].append(1.0)
                    else:
                        ys_pred_open_max[l].append(0.0)
        # print(ys_pred_open_max[0][j],ys_pred_open_max[1][j],ys_pred_open_max[2][j])

    retValue = []
    retValue_accuracy = []
    retValue_onehot = []

    sample_count = 0
    f_sample_count = 0
    f_sample_count_2 = 0
    for i in range(0, nb_outputs):
        # tmp_true=ys_true[:,i]
        tmp_true = ys_true[i]
        # print('True Lable '+str(i)+' :  ')
        # print(tmp_true)

        # tmp_pred=ys_pred[:,i]
        tmp_pred = ys_pred[i]
        # print('Pre Lable '+str(i)+' :  ')
        # print(tmp_pred)

        metrics_auc.reset_states()
        # print('index        ',i)
        # print('true        ',tmp_true.size)
        # print('pred        ',tmp_pred.size)
        metrics_auc.update_state(tmp_true, tmp_pred)
        retValue.append(metrics_auc.result().numpy())

        if len(ys_pred_open_max) > 0:

            c_true = tmp_true
            c_pred = ys_pred_open_max[i]

            sensitivity_c = 0
            specificity_c = 0

            for k in range(len(c_pred)):
                true_value = c_true[k]
                pred_value = c_pred[k]

                if pred_value == 1:
                    sample_count += 1

                if true_value == 1 and pred_value == 0:
                    f_sample_count += 1

                if true_value == 0 and pred_value == 1:
                    f_sample_count_2 += 1

                if true_value == 1 and pred_value == 1:
                    sensitivity_c += 1
                if true_value == 0 and pred_value == 0:
                    specificity_c += 1

            metrics_accuracy.reset_states()
            metrics_accuracy.update_state(tmp_true, ys_pred_open_max[i])
            retValue_onehot.append(
                [metrics_accuracy.result().numpy(), sensitivity_c, specificity_c, c_true.sum(), len(c_pred)])
            print('********', sample_count)
            print('&&&&&&&&&&&&&&&&&&', f_sample_count)
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', f_sample_count_2)

            metrics_accuracy.reset_states()
            metrics_accuracy.update_state(tmp_true, tmp_pred)
            retValue_accuracy.append(metrics_accuracy.result().numpy())

        else:
            metrics_accuracy.reset_states()
            metrics_accuracy.update_state(tmp_true, tmp_pred)
            retValue_accuracy.append(metrics_accuracy.result().numpy())

    if isReturnAccuracy:
        return [retValue, retValue_accuracy, retValue_onehot]
    else:
        return retValue

_custom_objects = {
    "CustomMultiLossLayer": CustomMultiLossLayer}

model_version = '210v'

def concate_result_2v(x):
    ret_value = []
    for i in range(len(x[0])):
        if i == 0 or i == 1:
            tmp_list = []
            for j in range(x[0][i].size):
                tmp_list.append(str(x[0][i][j], encoding="utf-8"))
            ret_value.append(np.array(tmp_list))
        elif i == 2:
            ret_value.append(x[0][i])
        else:
            ret_value.append(x[0][i].flatten())

    for n in range(1, len(x)):
        for i in range(len(x[n])):
            if i == 0 or i == 1:
                tmp_list = []
                for j in range(x[n][i].size):
                    tmp_list.append(str(x[n][i][j], encoding="utf-8"))
                ret_value[i] = np.concatenate((ret_value[i], np.array(tmp_list)), axis=0)
            elif i == 2:
                ret_value[i] = np.concatenate((ret_value[i], x[n][i]), axis=0)
            else:
                ret_value[i] = np.concatenate((ret_value[i], x[n][i].flatten()), axis=0)

    # print(ret_value)
    return ret_value


def concate_result_3v(x, muti_valu_index=3):
    ret_value = []
    for i in range(len(x[0])):
        if i == 0 or i == 1:
            tmp_list = []
            for j in range(x[0][i].size):
                tmp_list.append(str(x[0][i][j], encoding="utf-8"))
            ret_value.append(np.array(tmp_list))
        elif i >= 2 and i <= muti_valu_index:
            ret_value.append(x[0][i])
        else:
            ret_value.append(x[0][i].flatten())

    for n in range(1, len(x)):
        for i in range(len(x[n])):
            if i == 0 or i == 1:
                tmp_list = []
                for j in range(x[n][i].size):
                    tmp_list.append(str(x[n][i][j], encoding="utf-8"))
                ret_value[i] = np.concatenate((ret_value[i], np.array(tmp_list)), axis=0)
            elif i >= 2 and i <= muti_valu_index:
                ret_value[i] = np.concatenate((ret_value[i], x[n][i]), axis=0)
            else:
                ret_value[i] = np.concatenate((ret_value[i], x[n][i].flatten()), axis=0)

    # print(ret_value)
    return ret_value

def concate_result_5v(x, muti_valu_index=2):
    ret_value = []
    for i in range(len(x[0])):
        if i == 0 or i == 1:
            tmp_list = []
            for j in range(x[0][i].size):
                tmp_list.append(str(x[0][i][j], encoding="utf-8"))
            ret_value.append(np.array(tmp_list))
        elif i >= 2 and i <= muti_valu_index:
            ret_value.append(x[0][i])
        else:
            ret_value.append(x[0][i].flatten())

    for n in range(1, len(x)):
        for i in range(len(x[n])):
            if i == 0 or i == 1:
                tmp_list = []
                for j in range(x[n][i].size):
                    tmp_list.append(str(x[n][i][j], encoding="utf-8"))
                ret_value[i] = np.concatenate((ret_value[i], np.array(tmp_list)), axis=0)
            elif i >= 2 and i <= muti_valu_index:
                ret_value[i] = np.concatenate((ret_value[i], x[n][i]), axis=0)
            else:
                ret_value[i] = np.concatenate((ret_value[i], x[n][i].flatten()), axis=0)

    # print(ret_value)
    return ret_value


def get_methonds_2v(ys_true, ys_pred, rid, viscode, methond, model_path, p_weight=1, n_weight=1, save_flag=True,
                    is_test=False):
    data_len = ys_true[0].size

    data_dic = {}
    methonds_dict = {}
    for i in range(data_len):
        tmp_rid = rid[i]
        tmp_viscode = viscode[i]
        tmp_methond = methond[i]

        tmp_true = []
        for tmp_data_acm in ys_true:
            tmp_true.append(tmp_data_acm[i])

        tmp_pred = []
        for tmp_data_pred_acm in ys_pred:
            tmp_pred.append(tmp_data_pred_acm[i])

        key = str(tmp_rid) + '_' + str(tmp_viscode)

        if key not in data_dic:
            data_dic[key] = []
        tmp_data = data_dic[key]

        tmp_set = set()
        methond_str = ''
        # print(tmp_methond)
        for j in range(tmp_methond.size):
            methond_str += str(tmp_methond[j])
            if tmp_methond[j] == 1.0:
                tmp_set.add(j)
        # print('**********')
        # print(tmp_data)
        # print(tmp_set)
        # print(str(len(tmp_set)))
        # print(len(tmp_set))
        # print([len(tmp_set),tmp_set,tmp_true,tmp_pred,methond_str,tmp_methond])
        tmp_data.append([len(tmp_set), tmp_set, tmp_true, tmp_pred, methond_str, tmp_methond])
        data_dic[key] = tmp_data

    for key in data_dic:
        methonds = data_dic[key]
        methonds_sorted = sorted(methonds, key=lambda x: x[0])
        # key_array=key.split('-')
        # rid=key_array[0]
        #
        # viscode=key_array[1]

        # current_len=methonds_sorted[0][0]
        # next_index=0
        # current_methonds=methonds_sorted[0][1]
        # current_true = methonds_sorted[0][2]
        # current_pred_rate=methonds_sorted[0][3]

        for i in range(0, len(methonds_sorted)):
            current_methond = methonds_sorted[i]
            current_len = current_methond[0]
            current_methonds_set = current_methond[1]
            next_index = -1
            current_compared_len = current_len

            current_compared_rate = current_methond[3]
            current_methond_pre_transform = current_methond[5]
            current_key = key + '_' + current_methond[4]
            if is_test:
                methonds_dict[current_key] = [np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), current_methond[3]]
            else:
                for j in range(i + 1, len(methonds_sorted)):
                    next_tmp_methond = methonds_sorted[j]
                    next_len = next_tmp_methond[0]
                    next_methonds_set = next_tmp_methond[1]
                    # next_methonds_array=next_tmp_methond[5]

                    if next_len > current_compared_len:
                        if (current_compared_len != next_len) and (next_index != -1):
                            break

                        current_compared_len = next_len

                        tmp_next_true = next_tmp_methond[2]
                        tmp_next_pred = next_tmp_methond[3]

                        # 计算下一个诊断手段是否获得足够的增益。增益等于正例的加权平均增益加上负例的加权平均增益
                        sum_positive = 0
                        sum_negtive = 0

                        positive_count = 0
                        negtive_count = 0

                        for l in range(len(tmp_next_true)):
                            next_ture_rate = tmp_next_true[l]
                            next_pred_rate = tmp_next_pred[l]
                            current_compared_rate_correspond = current_compared_rate[l]

                            if next_ture_rate >= 0.5:
                                positive_count += 1
                                sum_positive += (next_pred_rate - current_compared_rate_correspond)
                            else:
                                negtive_count += 1
                                sum_negtive += (current_compared_rate_correspond - next_pred_rate)

                        if (p_weight * (sum_positive / positive_count) + n_weight * (sum_negtive / negtive_count)) > 0:
                            if current_methonds_set.issubset(next_methonds_set):
                                next_index = j
                                current_compared_rate = tmp_next_pred
                if next_index == -1:
                    methonds_dict[current_key] = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), current_methond[3]]
                else:
                    # methonds_sorted[next_index]
                    transform_methond = methonds_sorted[next_index][5]
                    clear_transform = np.zeros(13, dtype=int)
                    for k in range(transform_methond.size):
                        if transform_methond[k] == 1 and current_methond_pre_transform[k] == 0:
                            clear_transform[k] = 1
                            statistics_transform[k] = statistics_transform[k] + 1
                    methonds_dict[current_key] = [clear_transform, current_methond[3]]

    if save_flag:
        if os.path.exists(model_path + '_methonds.npy'):
            os.remove(model_path + '_methonds.npy')
        np.save(model_path + '_methonds.npy', methonds_dict)
    return methonds_dict

def get_methond_batch(rid, viscode, methond, model_path, dic_type=0, methond_trans_kind_count=12, pred_kind_count=3,
                      rid_is_array=False):
    # print(rid.shape)
    # print(viscode.shape)
    # print(methond.shape)
    data_dic = ''
    ret_value = []
    for i in range(methond_trans_kind_count + pred_kind_count):  # 记录下一步诊断手段，以及当前的诊断类别概率
        ret_value.append([])
        # [[]] * ()

    model_path = model_path + '_methonds.npy'
    if dic_type == 0:
        if len(acm_methond_dic_train) < 1:
            data_dic = np.load(model_path, allow_pickle=True).item()
            print('######################################')
            print(len(data_dic))
            # print(data_dic)
            # acm_methond_dic_train = data_dic
            acm_methond_dic_train.update(data_dic)
        else:
            data_dic = acm_methond_dic_train
    elif dic_type == 1:
        if len(acm_methond_dic_eval) < 1:
            data_dic = np.load(model_path, allow_pickle=True).item()
            # acm_methond_dic_eval = data_dic
            acm_methond_dic_eval.update(data_dic)
        else:
            data_dic = acm_methond_dic_eval

    if not rid_is_array:
        rid = rid.numpy()
        viscode = viscode.numpy()

    isContainData = False
    for i in range(rid.size):
        rid_str = str(rid[i], encoding='utf-8')
        viscode_str = str(viscode[i], encoding='utf-8')

        methond_str = ''
        # print(tmp_methond)
        for j in range(methond[i].size):
            methond_str += str(methond[i][j])

        current_key = rid_str + '_' + viscode_str + '_' + methond_str

        if current_key in data_dic:
            isContainData = True
            methond_trans_data = data_dic[current_key][0]
            pred_rate = data_dic[current_key][1]

            for l in range(1, methond_trans_data.size):
                ret_value[l - 1].append(methond_trans_data[l])

            for l in range(len(pred_rate)):
                ret_value[l + methond_trans_data.size - 1].append(pred_rate[l])

        else:
            print('数据不在字典内', current_key)

    final_ret_value = []
    if isContainData:
        for ele in ret_value:
            # print(len(ele))
            final_ret_value.append(np.array(ele))

    return final_ret_value

def methond_simplified(x):
    # print(x.shape)
    tmp_method_list = []
    for j in range(len(x)):  # bath
        # print('Batch 中的第： ' + str(j) + ' 个样本！')
        tmp_method = np.zeros(13, dtype=int)
        for k in range(len(x[j])):  # 第一层
            # print(x[j][k][0:60])
            if np.sum(x[j][k][0:10]) == 0:
                # print('样本中的第： ' + str(k) + ' 个步长！')
                # print(x[j][k][10:60])
                for l in range(10, 60):
                    if x[j][k][l] == 1.0:
                        index = l - 10
                        index = math.floor(index / 2)
                        tmp_method[index] = 1
        # print(tmp_method)
        tmp_method_list.append(tmp_method)
    return np.array(tmp_method_list)


def get_index(next_methond, c_index, methonds_sorted):
    c_m_set = methonds_sorted[c_index][5]

    for k in range(c_index + 1, len(methonds_sorted)):
        tmp_next_methond = methonds_sorted[k][3]
        n_m_set = methonds_sorted[k][5]

        if c_m_set.issubset(n_m_set):
            tmp_flag = True
            for l in range(len(next_methond)):
                if next_methond[l] != tmp_next_methond[l]:
                    tmp_flag = False
            if tmp_flag:
                return k

    return -1


# trans_methond_count=0
def get_next_methond(methonds_sorted, current_index, next_pred_methond, methond_high=0.8, methond_low=0.3):
    current_methond = methonds_sorted[current_index][3]
    # current_methond_set=methonds_sorted[current_index][5]

    next_trans_methond = []
    next_methond = []
    next_methond.append(1)
    for i in range(len(next_pred_methond)):
        pred_value = next_pred_methond[i]
        methond_value = 0
        if pred_value > methond_high and current_methond[i + 1] == 0:
            methond_value = 1
        next_trans_methond.append(methond_value)

    if sum(next_trans_methond) > 0:  # 按照预测值得到下一步的检测手段
        for j in range(len(next_trans_methond)):
            trans_value = next_trans_methond[j]
            c_methond_value = current_methond[j + 1]

            if trans_value == 1 or c_methond_value == 1:
                next_methond.append(1)
            else:
                next_methond.append(0)
        k = get_index(next_methond, current_index, methonds_sorted)
        if k != -1:
            return k

    trans_methond_count[0] += 1

    next_methond.clear()
    for l in range(current_methond.size):  # 从代价轻的开始往后补充检测手段,
        tmp_value = current_methond[l]
        if tmp_value == 1:
            next_methond.append(tmp_value)
        else:
            # next_methond.append(1)
            tmp_next_methond = copy.copy(next_methond)
            tmp_next_methond.append(1)
            for k in range(l + 1, current_methond.size):
                tmp_next_methond.append(current_methond[k])

            tmp_index = get_index(tmp_next_methond, current_index, methonds_sorted)
            if tmp_index != -1:
                return tmp_index
            else:
                trans_methond_count[0] += 1
                next_methond.append(tmp_value)

    return -1

# 闭集
def get_modify_prediction_4v(rid, viscode, methond, ys_true, ys_pred, ys_pred_next_methond, completeness=0.6,
                             pred_high=[0.8, 0.8], pred_low=[0.3, 0.3], methond_high=0.8, methond_low=0.3,
                             iscontaincogforall=False):
    data_dic = {}
    data_len = rid.size

    m_ys_ture = []
    m_ys_pred = []

    for i in range(len(ys_true)):
        m_ys_ture.append([])
        m_ys_pred.append([])

    m_rid = []
    m_viscode = []
    m_methond = []
    m_path = []

    for i in range(data_len):
        tmp_rid = rid[i]
        tmp_viscode = viscode[i]
        tmp_methond = methond[i]

        tmp_true = []
        for tmp_data_acm in ys_true:
            tmp_true.append(tmp_data_acm[i])

        tmp_pred = []
        for tmp_data_pred_acm in ys_pred:
            tmp_pred.append(tmp_data_pred_acm[i])

        tmp_pred_methond = []
        for tmp_pred_acm_next_methond in ys_pred_next_methond:
            tmp_pred_methond.append(tmp_pred_acm_next_methond[i])

        key = str(tmp_rid) + '_' + str(tmp_viscode)

        if key not in data_dic:
            data_dic[key] = []
        tmp_data = data_dic[key]

        tmp_set = set()
        # print(tmp_methond)
        for j in range(tmp_methond.size):
            if tmp_methond[j] == 1.0:
                tmp_set.add(j)
        tmp_data.append([len(tmp_set), tmp_true, tmp_pred, tmp_methond, tmp_pred_methond, tmp_set])
        data_dic[key] = tmp_data

    viscode_count = 0

    # print(data_dic)

    # trans_methond_count=0
    # global trans_methond_count

    lack_methond_total = {}

    for key in data_dic:
        key_list = key.split('_')
        tmp_rid = key_list[0]
        tmp_viscode = key_list[1]
        viscode_count += 1
        methonds = data_dic[key]
        methonds_sorted = sorted(methonds, key=lambda x: x[0])

        if iscontaincogforall:
            # 检查是否包含认知检测
            clear_methonds_sorted = []
            isContainCog = False
            for tmp_methonds_el in methonds_sorted:
                if 1 in tmp_methonds_el[5]:
                    isContainCog = True
                    break

            if isContainCog:
                for tmp_methonds_el in methonds_sorted:
                    if (0 in tmp_methonds_el[5]) and (1 in tmp_methonds_el[5]):
                        clear_methonds_sorted.append(tmp_methonds_el)
                methonds_sorted = clear_methonds_sorted

        diagnosis_path = []

        i = 0

        isHasAnswer = False
        while (i < len(methonds_sorted)):
            current_methond = methonds_sorted[i]
            sample_true = current_methond[1]
            sample_pred = current_methond[2]
            methond = current_methond[3]
            next_pred_methond = current_methond[4]
            diagnosis_path.append(methond)

            # diagnosis_path.append(methond)

            p_count = 0
            n_count = 0
            for k in range(len(sample_pred)):
                el = sample_pred[k]
                if el > pred_high[k]:
                    p_count += 1
                elif el < pred_low[k]:
                    n_count += 1

            if (p_count > 0 and (p_count + n_count) / len(sample_pred) > completeness) or i == (
                    len(methonds_sorted) - 1):
                for l in range(len(sample_true)):
                    if l < len(sample_pred):
                        m_ys_pred[l].append(sample_pred[l])
                    else:
                        m_ys_pred[l].append(0)
                    m_ys_ture[l].append(sample_true[l])
                m_rid.append(tmp_rid)
                m_viscode.append(tmp_viscode)
                m_methond.append(methond)
                m_path.append(diagnosis_path)

                # if methonds_sorted[0][1][2] == 1:
                #    print('*****************************')
                #    for el in methonds_sorted:
                #        print(tmp_rid, tmp_viscode, el[0], el[1], el[2], el[3])
                #    print('*****************************')
                print([tmp_rid, tmp_viscode, methond, sample_true, sample_pred])

                isHasAnswer = True
                break
            else:
                i = get_next_methond(methonds_sorted, i, next_pred_methond, methond_high=methond_high,
                                     methond_low=methond_low)
                if i == -1:
                    break

        if not isHasAnswer:
            last_methond = methonds_sorted[-1]
            sample_true = last_methond[1]
            sample_pred = last_methond[2]
            methond = last_methond[3]
            diagnosis_path.append(methond)

            for l in range(len(sample_true)):
                if l < len(sample_pred):
                    m_ys_pred[l].append(sample_pred[l])
                else:
                    m_ys_pred[l].append(0)
                m_ys_ture[l].append(sample_true[l])
            m_rid.append(tmp_rid)
            m_viscode.append(tmp_viscode)
            m_methond.append(methond)
            m_path.append(diagnosis_path)
        # print('########',m_ys_pred)
    return m_rid, m_viscode, m_methond, m_ys_ture, get_one_hot_2v(m_ys_pred, len(m_ys_pred[0])), m_path

# 添加数据标签 就是矩形上面的数值
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
        rect.set_edgecolor('white')


def draw_picture(picture_type, x, y, title, xlabel, ylabel):
    font1 = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 15,
             }
    font2 = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 10,
             }
    plt.cla()
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel(xlabel, font1)
    plt.ylabel(ylabel, font1)
    plt.legend(prop=font2)
    plt.tick_params(labelsize=13)

    if picture_type == 0:
        methond_name = ['Base', 'Cog', 'CE', 'Neur', 'FB', 'PE', 'Blood', 'Urine', 'MRI', 'FDG', 'AV45', 'Gene', 'CSF']
        methond_y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        plt.yticks(methond_y, methond_name)
        plt.scatter(x, y, s=8)
    elif picture_type == 1:
        bar_data = plt.bar(y, x)
        add_labels(bar_data)
    elif picture_type == 2:
        y_xiv = []
        for i in range(len(x)):
            y_xiv.append(i + 1)
        plt.bar(y_xiv, x)
    plt.show()


def pain(rid, viscode, methond, ys_true):
    scatter = [[], []]
    bl_scatter = [[], []]
    other_scatter = [[], []]

    scatter_ad = [[], []]
    bl_scatter_ad = [[], []]
    other_scatter_ad = [[], []]

    scatter_cn = [[], []]
    bl_scatter_cn = [[], []]
    other_scatter_cn = [[], []]

    scatter_mci = [[], []]
    bl_scatter_mci = [[], []]
    other_scatter_mci = [[], []]

    scatter_smc = [[], []]
    bl_scatter_smc = [[], []]
    other_scatter_smc = [[], []]

    scatter_unknown = [[], []]
    bl_scatter_unknown = [[], []]
    other_scatter_unknown = [[], []]

    hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bl_hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    other_hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    hist_ad = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bl_hist_ad = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    other_hist_ad = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    hist_cn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bl_hist_cn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    other_hist_cn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    hist_mci = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bl_hist_mci = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    other_hist_mci = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    hist_smc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bl_hist_smc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    other_hist_smc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    hist_unknown = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bl_hist_unknown = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    other_hist_unknown = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    user = []
    bl_user = []
    other_user = []

    ad_user = []
    bl_ad_user = []
    other_ad_user = []

    cn_user = []
    bl_cn_user = []
    other_cn_user = []

    mci_user = []
    bl_mci_user = []
    other_mci_user = []

    smc_user = []
    bl_smc_user = []
    other_smc_user = []

    for i in range(len(rid)):
        tmp_viscode = viscode[i]
        tmp_methond = methond[i]
        user.append(sum(tmp_methond))

        dignose_type = 0
        if ys_true[0][i] == 1:
            ad_user.append(sum(tmp_methond))
            if tmp_viscode == 'bl':
                bl_ad_user.append(sum(tmp_methond))
            else:
                other_ad_user.append(sum(tmp_methond))
        elif ys_true[1][i] == 1:
            dignose_type = 1
            cn_user.append(sum(tmp_methond))
            if tmp_viscode == 'bl':
                bl_cn_user.append(sum(tmp_methond))
            else:
                other_cn_user.append(sum(tmp_methond))
        elif ys_true[2][i] == 1:
            dignose_type = 2
            mci_user.append(sum(tmp_methond))
            if tmp_viscode == 'bl':
                bl_mci_user.append(sum(tmp_methond))
            else:
                other_mci_user.append(sum(tmp_methond))
        elif ys_true[3][i] == 1:
            dignose_type = 3
            smc_user.append(sum(tmp_methond))
            if tmp_viscode == 'bl':
                bl_smc_user.append(sum(tmp_methond))
            else:
                other_smc_user.append(sum(tmp_methond))

        if tmp_viscode == 'bl':
            bl_user.append(sum(tmp_methond))
        else:
            other_user.append(sum(tmp_methond))

        for j in range(tmp_methond.size):
            if tmp_methond[j] == 1:
                scatter[0].append(i)
                scatter[1].append(j + 1)

                hist[j] = hist[j] + 1

                if tmp_viscode == 'bl':
                    bl_scatter[0].append(i)
                    bl_scatter[1].append(j + 1)

                    bl_hist[j] = bl_hist[j] + 1
                else:
                    other_scatter[0].append(i)
                    other_scatter[1].append(j + 1)

                    other_hist[j] = other_hist[j] + 1

                if dignose_type == 0:
                    scatter_ad[0].append(i)
                    scatter_ad[1].append(j + 1)

                    hist_ad[j] = hist_ad[j] + 1

                    if tmp_viscode == 'bl':
                        bl_scatter_ad[0].append(i)
                        bl_scatter_ad[1].append(j + 1)

                        bl_hist_ad[j] = bl_hist_ad[j] + 1
                    else:
                        other_scatter_ad[0].append(i)
                        other_scatter_ad[1].append(j + 1)

                        other_hist_ad[j] = other_hist_ad[j] + 1
                elif dignose_type == 1:
                    scatter_cn[0].append(i)
                    scatter_cn[1].append(j + 1)

                    hist_cn[j] = hist_cn[j] + 1

                    if tmp_viscode == 'bl':
                        bl_scatter_cn[0].append(i)
                        bl_scatter_cn[1].append(j + 1)

                        bl_hist_cn[j] = bl_hist_cn[j] + 1
                    else:
                        other_scatter_cn[0].append(i)
                        other_scatter_cn[1].append(j + 1)

                        other_hist_cn[j] = other_hist_cn[j] + 1
                elif dignose_type == 2:
                    scatter_mci[0].append(i)
                    scatter_mci[1].append(j + 1)
                    scatter_unknown[0].append(i)
                    scatter_unknown[1].append(j + 1)

                    hist_mci[j] = hist_mci[j] + 1
                    hist_unknown[j] = hist_unknown[j] + 1

                    if tmp_viscode == 'bl':
                        bl_scatter_mci[0].append(i)
                        bl_scatter_mci[1].append(j + 1)

                        bl_scatter_unknown[0].append(i)
                        bl_scatter_unknown[1].append(j + 1)

                        bl_hist_mci[j] = bl_hist_mci[j] + 1
                        bl_hist_unknown[j] = bl_hist_unknown[j] + 1
                    else:
                        other_scatter_mci[0].append(i)
                        other_scatter_mci[1].append(j + 1)

                        other_scatter_unknown[0].append(i)
                        other_scatter_unknown[1].append(j + 1)

                        other_hist_mci[j] = other_hist_mci[j] + 1
                        other_hist_unknown[j] = other_hist_unknown[j] + 1
                elif dignose_type == 3:
                    scatter_smc[0].append(i)
                    scatter_smc[1].append(j + 1)

                    scatter_unknown[0].append(i)
                    scatter_unknown[1].append(j + 1)

                    hist_smc[j] = hist_smc[j] + 1
                    hist_unknown[j] = hist_unknown[j] + 1

                    if tmp_viscode == 'bl':
                        bl_scatter_smc[0].append(i)
                        bl_scatter_smc[1].append(j + 1)

                        bl_scatter_unknown[0].append(i)
                        bl_scatter_unknown[1].append(j + 1)

                        bl_hist_smc[j] = bl_hist_smc[j] + 1
                        bl_hist_unknown[j] = bl_hist_unknown[j] + 1
                    else:
                        other_scatter_smc[0].append(i)
                        other_scatter_smc[1].append(j + 1)

                        other_scatter_unknown[0].append(i)
                        other_scatter_unknown[1].append(j + 1)

                        other_hist_smc[j] = other_hist_smc[j] + 1
                        other_hist_unknown[j] = other_hist_unknown[j] + 1

    while (True):
        draw_type = eval(input('输入图表模式：'))
        if draw_type == 0:
            break
        elif draw_type == -1:
            draw_picture(0, scatter_unknown[0], scatter_unknown[1], 'The examination for Unknown subject diagnosis',
                         'Subject ID', 'Examination')
        elif draw_type == -2:
            draw_picture(0, bl_scatter_unknown[0], bl_scatter_unknown[1],
                         'The examination for Unknown subject baseline diagnosis', 'Subject ID', 'Examination')
        elif draw_type == -3:
            draw_picture(0, other_scatter_unknown[0], other_scatter_unknown[1],
                         'The examination for Unknown subject non baseline diagnosis', 'Subject ID', 'Examination')

        elif draw_type == 1:
            draw_picture(0, scatter[0], scatter[1], 'The examination for diagnosis', 'Subject ID', 'Examination')
        elif draw_type == 2:
            draw_picture(0, bl_scatter[0], bl_scatter[1], 'The examination for baseline diagnosis', 'Subject ID',
                         'Examination')
        elif draw_type == 3:
            draw_picture(0, other_scatter[0], other_scatter[1], 'The examination for non baseline diagnosis',
                         'Subject ID', 'Examination')
        elif draw_type == 4:
            draw_picture(0, scatter_ad[0], scatter_ad[1], 'The examination for AD subject diagnosis', 'Subject ID',
                         'Examination')
        elif draw_type == 5:
            draw_picture(0, bl_scatter_ad[0], bl_scatter_ad[1], 'The examination for AD subject baseline diagnosis',
                         'Subject ID', 'Examination')
        elif draw_type == 6:
            draw_picture(0, other_scatter_ad[0], other_scatter_ad[1],
                         'The examination for AD subject non baseline diagnosis', 'Subject ID', 'Examination')
        elif draw_type == 7:
            draw_picture(0, scatter_cn[0], scatter_cn[1], 'The examination for CN subject diagnosis', 'Subject ID',
                         'Examination')
        elif draw_type == 8:
            draw_picture(0, bl_scatter_cn[0], bl_scatter_cn[1], 'The examination for CN subject baseline diagnosis',
                         'Subject ID', 'Examination')
        elif draw_type == 9:
            draw_picture(0, other_scatter_cn[0], other_scatter_cn[1],
                         'The examination for CN subject non baseline diagnosis', 'Subject ID', 'Examination')
        elif draw_type == 10:
            draw_picture(0, scatter_mci[0], scatter_mci[1], 'The examination for MCI subject diagnosis', 'Subject ID',
                         'Examination')
        elif draw_type == 11:
            draw_picture(0, bl_scatter_mci[0], bl_scatter_mci[1], 'The examination for MCI subject baseline diagnosis',
                         'Subject ID', 'Examination')
        elif draw_type == 12:
            draw_picture(0, other_scatter_mci[0], other_scatter_mci[1],
                         'The examination for MCI subject non baseline diagnosis', 'Subject ID', 'Examination')

        elif draw_type == 13:
            draw_picture(0, scatter_smc[0], scatter_smc[1], 'The examination for SMC subject diagnosis', 'Subject ID',
                         'Examination')
        elif draw_type == 14:
            draw_picture(0, bl_scatter_smc[0], bl_scatter_smc[1], 'The examination for SMC subject baseline diagnosis',
                         'Subject ID', 'Examination')
        elif draw_type == 15:
            draw_picture(0, other_scatter_smc[0], other_scatter_smc[1],
                         'The examination for SMC subject non baseline diagnosis', 'Subject ID', 'Examination')


        elif draw_type == -4:
            draw_picture(1, hist_unknown, ['Base', 'Cog', 'CE', 'Neur', 'FB', 'PE', 'Blood',
                                           'Urine', 'MRI', 'FDG', 'AV45', 'Gene', 'CSF'],
                         'The methonds number of Unknown diagnosis', 'Methond', 'Subject')
        elif draw_type == -5:
            draw_picture(1, bl_hist_unknown,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of baseline Unknown diagnosis', 'Methond', 'Subject')
        elif draw_type == -6:
            draw_picture(1, other_hist_unknown,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of non baseline Unknown diagnosis', 'Methond', 'Subject')
        elif draw_type == 16:
            draw_picture(1, hist, ['Base', 'Cog', 'CE', 'Neur', 'FB', 'PE', 'Blood',
                                   'Urine', 'MRI', 'FDG', 'AV45', 'Gene', 'CSF'], 'The methonds number of diagnosis',
                         'Methond', 'Subject')
        elif draw_type == 17:
            draw_picture(1, bl_hist, ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                                      'Physical examination', 'Blood',
                                      'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of baseline diagnosis', 'Methond', 'Subject')
        elif draw_type == 18:
            draw_picture(1, other_hist,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of non baseline diagnosis', 'Methond', 'Subject')
        elif draw_type == 19:
            draw_picture(1, hist_ad, ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                                      'Physical examination', 'Blood',
                                      'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of AD diagnosis', 'Methond', 'Subject')
        elif draw_type == 20:
            draw_picture(1, bl_hist_ad,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of baseline AD diagnosis', 'Methond', 'Subject')
        elif draw_type == 21:
            draw_picture(1, other_hist_ad,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of non baseline AD diagnosis', 'Methond', 'Subject')
        elif draw_type == 22:
            draw_picture(1, hist_cn, ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                                      'Physical examination', 'Blood',
                                      'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of CN diagnosis', 'Methond', 'Subject')
        elif draw_type == 23:
            draw_picture(1, bl_hist_cn,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of baseline CN diagnosis', 'Methond', 'Subject')
        elif draw_type == 24:
            draw_picture(1, other_hist_cn,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of non baseline CN diagnosis', 'Methond', 'Subject')
        elif draw_type == 25:
            draw_picture(1, hist_mci, ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                                       'Physical examination', 'Blood',
                                       'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of MCI diagnosis', 'Methond', 'Subject')
        elif draw_type == 26:
            draw_picture(1, bl_hist_mci,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of baseline MCI diagnosis', 'Methond', 'Subject')
        elif draw_type == 27:
            draw_picture(1, other_hist_mci,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of non baseline MCI diagnosis', 'Methond', 'Subject')

        elif draw_type == 28:
            draw_picture(1, hist_smc, ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                                       'Physical examination', 'Blood',
                                       'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of SMC diagnosis', 'Methond', 'Subject')
        elif draw_type == 29:
            draw_picture(1, bl_hist_smc,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of baseline SMC diagnosis', 'Methond', 'Subject')
        elif draw_type == 30:
            draw_picture(1, other_hist_smc,
                         ['Base', 'Cognition', 'Cognition examination', 'Neurology', 'Function behaviour',
                          'Physical examination', 'Blood',
                          'Urine', 'MRI', 'PET 18F-FDG', 'PET 18F-AV45', 'Gene', 'CSF'],
                         'The methonds number of non baseline SMC diagnosis', 'Methond', 'Subject')

        elif draw_type == 31:
            draw_picture(2, user, [], 'The methonds number of subject diagnosis', 'Methond number', 'Subject')
        elif draw_type == 32:
            draw_picture(2, bl_user, [], 'The methonds number of subject baseline diagnosis', 'Methond number',
                         'Subject')
        elif draw_type == 33:
            draw_picture(2, other_user, [], 'The methonds number of subject non baseline diagnosis', 'Methond number',
                         'Subject')
        elif draw_type == 34:
            draw_picture(2, ad_user, [], 'The methonds number of subject AD diagnosis', 'Methond number', 'Subject')
        elif draw_type == 35:
            draw_picture(2, bl_ad_user, [], 'The methonds number of subject baseline AD diagnosis', 'Methond number',
                         'Subject')
        elif draw_type == 36:
            draw_picture(2, other_ad_user, [], 'The methonds number of subject non baseline AD diagnosis',
                         'Methond number', 'Subject')
        elif draw_type == 37:
            draw_picture(2, cn_user, [], 'The methonds number of subject CN diagnosis', 'Methond number', 'Subject')
        elif draw_type == 38:
            draw_picture(2, bl_cn_user, [], 'The methonds number of subject baseline CN diagnosis', 'Methond number',
                         'Subject')
        elif draw_type == 39:
            draw_picture(2, other_cn_user, [], 'The methonds number of subject non baseline CN diagnosis',
                         'Methond number', 'Subject')
        elif draw_type == 40:
            draw_picture(2, mci_user, [], 'The methonds number of subject MCI diagnosis', 'Methond number', 'Subject')
        elif draw_type == 41:
            draw_picture(2, bl_mci_user, [], 'The methonds number of subject baseline MCI diagnosis', 'Methond number',
                         'Subject')
        elif draw_type == 42:
            draw_picture(2, other_mci_user, [], 'The methonds number of subject non baseline MCI diagnosis',
                         'Methond number', 'Subject')

        elif draw_type == 43:
            draw_picture(2, smc_user, [], 'The methonds number of subject SMC diagnosis', 'Methond number', 'Subject')
        elif draw_type == 44:
            draw_picture(2, bl_smc_user, [], 'The methonds number of subject baseline SMC diagnosis', 'Methond number',
                         'Subject')
        elif draw_type == 45:
            draw_picture(2, other_smc_user, [], 'The methonds number of subject non baseline SMC diagnosis',
                         'Methond number', 'Subject')


def compute_distance(a, b, metric_type='cosine'):
    return paired_distances(a, b, metric=metric_type, n_jobs=1)


def compute_distance_2v(a, b, metric_type='cosine', top_one=True):
    dis_array = []
    if not top_one:
        for i in range(b.shape[0]):
            dis_array.append(paired_distances(a.reshape(1, -1), b[i].reshape(1, -1), metric=metric_type, n_jobs=1))
        if len(dis_array) < 2:
            return dis_array[0]
        elif len(dis_array) < 3:
            num1 = min(dis_array)
            dis_array.remove(num1)
            num2 = min(dis_array)
            return (num1 + num2) / 2
        else:
            num1 = min(dis_array)
            dis_array.remove(num1)
            num2 = min(dis_array)
            dis_array.remove(num2)
            num3 = min(dis_array)

            return (num1 + num2 + num3) / 3
    else:
        for i in range(b.shape[0]):
            dis_array.append(paired_distances(a.reshape(1, -1), b[i].reshape(1, -1), metric=metric_type, n_jobs=1))
        return min(dis_array)


def query_weibull(label, weibull_model):
    return [
        [weibull_model[label]['mean_vec']],
        [weibull_model[label]['distances']],
        [weibull_model[label]['weibull_model']]
    ]


def weibull_fit_tails_2v(av_map, tail_size=2000, metric_type='cosine', center_path=''):
    weibull_model = {}
    labels = av_map.keys()

    if not os.path.exists(center_path + 'ad_centers.npy'):
        ad_class_av = av_map[0]
        cn_class_av = av_map[1]

        mbk_ad = MiniBatchKMeans(init='k-means++', n_clusters=10, max_iter=150, random_state=0, batch_size=512,
                                 n_init=10, max_no_improvement=10, verbose=0)
        mbk_ad.fit(ad_class_av)

        mbk_cn = MiniBatchKMeans(init='k-means++', n_clusters=10, max_iter=150, random_state=0, batch_size=512,
                                 n_init=10, max_no_improvement=10, verbose=0)
        mbk_cn.fit(cn_class_av)

        means_ad = mbk_ad.cluster_centers_
        means_cn = mbk_cn.cluster_centers_
        np.save(center_path + 'ad_centers.npy', means_ad)
        np.save(center_path + 'cn_centers.npy', means_cn)

    for label in labels:
        print(f'EVT fitting for label {label}')
        weibull_model[label] = {}

        class_av = av_map[label]
        class_mav = np.mean(class_av, axis=0, keepdims=True)

        av_distance = np.zeros((1, class_av.shape[0]))

        if center_path == '':
            for i in range(class_av.shape[0]):
                av_distance[0, i] = compute_distance(class_av[i, :].reshape(1, -1), class_mav, metric_type=metric_type)
        else:
            means_ad = np.load(center_path + 'ad_centers.npy')
            means_cn = np.load(center_path + 'cn_centers.npy')

            if label == 0:  # ad
                for i in range(class_av.shape[0]):
                    tmp_ad_distance = compute_distance_2v(class_av[i, :], means_ad, metric_type=metric_type)
                    tmp_cn_distance = compute_distance_2v(class_av[i, :], means_cn, metric_type=metric_type)
                    av_distance[0, i] = math.sqrt(math.pow(tmp_ad_distance, 2) + math.pow(1 - tmp_cn_distance, 2))
            elif label == 1:  # cn
                for i in range(class_av.shape[0]):
                    tmp_ad_distance = compute_distance_2v(class_av[i, :], means_ad, metric_type=metric_type)
                    tmp_cn_distance = compute_distance_2v(class_av[i, :], means_cn, metric_type=metric_type)
                    av_distance[0, i] = math.sqrt(math.pow(1 - tmp_ad_distance, 2) + math.pow(tmp_cn_distance, 2))
        weibull_model[label]['mean_vec'] = class_mav
        weibull_model[label]['distances'] = av_distance

        mr = libmr.MR()

        tail_size_fix = min(tail_size, av_distance.shape[1])
        tails_to_fit = sorted(av_distance[0, :])[-tail_size_fix:]
        mr.fit_high(tails_to_fit, tail_size_fix)

        weibull_model[label]['weibull_model'] = mr

    return weibull_model


def weibull_fit_2v(model_path, score, prob, y, center_path=''):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    print('Model do not exists !      ', model_path)
    predicted_y = np.argmax(prob, axis=1)

    labels = np.unique(y)
    av_map = {}

    for label in labels:
        av_map[label] = score[(y == label) & (predicted_y == y), :]

    model = weibull_fit_tails_2v(av_map, tail_size=800, center_path=center_path)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model


def compute_openmax_probability(openmax_score, openmax_score_unknown, labels):
    exp_scores = []
    for label_index, label in enumerate(labels):
        exp_scores += [np.exp(openmax_score[label_index])]

    total_denominator = np.sum(np.exp(openmax_score)) + np.exp(np.sum(openmax_score_unknown))
    prob_scores = np.array(exp_scores) / total_denominator
    prob_unknown = np.exp(np.sum(openmax_score_unknown)) / total_denominator

    return prob_scores.tolist() + [prob_unknown]


def compute_softmax_probability(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def recalibrate_scores_4v(activation_vector, weibull_model, labels, alpha_rank=10, isLastLayer=True, real_acv=[],
                          score_cal_type=1, modify=0, center_path='', means_ad=[], means_cn=[], weibull_ad_thr=0,
                          weibull_cn_thr=0, savepath='', ad=0, cn=0):
    # print('ad_thr  ',weibull_ad_thr)
    # print('cn_thr  ',weibull_cn_thr)
    ranked_list = activation_vector.argsort().ravel()[::-1]  # 升序排序，倒置，返回的是一个降序排序的下标，原数组不变
    alpha_weights = [((alpha_rank + 1) - i) / float(alpha_rank) for i in range(1, alpha_rank + 1)]
    ranked_alpha = np.zeros((len(labels)))

    for i in range(alpha_rank):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Now recalibrate score for each class to include probability of unknown
    openmax_score = []
    openmax_score_unknown = []

    openmax_wscoe = []

    ret_str = ''
    if savepath != '':
        for i in range(activation_vector.size):
            ret_str += str(activation_vector[i]) + ','

    for label_index, label in enumerate(labels):
        if center_path == '':
            # get distance between current channel and mean vector
            weibull = query_weibull(label, weibull_model)
            if isLastLayer:
                av_distance = compute_distance(activation_vector.reshape(1, -1), weibull[0][0])
            else:
                av_distance = compute_distance(real_acv.reshape(1, -1), weibull[0][0])
            # obtain w_score for the distance and compute probability of the distance being unknown wrt to mean training vector
            wscore = weibull[2][0].w_score(av_distance)
            modified_score = activation_vector[label_index] * (1 - wscore * ranked_alpha[label_index])
            openmax_score += [modified_score]
            openmax_score_unknown += [activation_vector[label_index] - modified_score]
        else:
            # means_ad = np.load(center_path + 'ad_centers.npy')
            # means_cn = np.load(center_path + 'cn_centers.npy')
            weibull = query_weibull(label, weibull_model)
            av_distance = 0
            wscore = 0
            if label == 0:  # ad
                if isLastLayer:
                    tmp_ad_distance = compute_distance_2v(activation_vector, means_ad)
                    tmp_cn_distance = compute_distance_2v(activation_vector, means_cn)
                    av_distance = math.sqrt(math.pow(tmp_ad_distance, 2) + math.pow(1 - tmp_cn_distance, 2))
                else:
                    tmp_ad_distance = compute_distance_2v(real_acv, means_ad)
                    tmp_cn_distance = compute_distance_2v(real_acv, means_cn)
                    av_distance = math.sqrt(math.pow(tmp_ad_distance, 2) + math.pow(1 - tmp_cn_distance, 2))

                ad_gap = av_distance - weibull_ad_thr

                if ad_gap <= 0:
                    wscore = 0
                else:
                    wscore = ad_gap / weibull_ad_thr
                    if wscore > 1:
                        wscore == 1

                if savepath != '':
                    ret_str += str(av_distance) + ','
                    ret_str += str(ad_gap) + ','
                    ret_str += str(wscore) + ','

                openmax_wscoe.append(wscore)

            elif label == 1:  # cn
                if isLastLayer:
                    tmp_ad_distance = compute_distance_2v(activation_vector, means_ad)
                    tmp_cn_distance = compute_distance_2v(activation_vector, means_cn)
                    av_distance = math.sqrt(math.pow(1 - tmp_ad_distance, 2) + math.pow(tmp_cn_distance, 2))
                else:
                    tmp_ad_distance = compute_distance_2v(real_acv, means_ad)
                    tmp_cn_distance = compute_distance_2v(real_acv, means_cn)
                    av_distance = math.sqrt(math.pow(1 - tmp_ad_distance, 2) + math.pow(tmp_cn_distance, 2))

                cn_gap = av_distance - weibull_cn_thr

                if cn_gap <= 0:
                    wscore = 0
                else:
                    wscore = cn_gap / weibull_cn_thr
                    if wscore > 1:
                        wscore == 1

                if savepath != '':
                    ret_str += str(av_distance) + ','
                    ret_str += str(cn_gap) + ','
                    ret_str += str(wscore) + ','

                openmax_wscoe.append(wscore)

            if av_distance == 0:
                print('^^^^^^^^^^   ERROR')

            model_wscore = weibull[2][0].w_score(av_distance)

            modified_score = 0
            if score_cal_type == 1:
                modified_score = activation_vector[label_index] * (1 - model_wscore * ranked_alpha[label_index])
                openmax_score += [modified_score]
                openmax_score_unknown += [activation_vector[label_index] - modified_score]
            elif score_cal_type == 2:
                modified_score = activation_vector[label_index] * (1 - wscore * ranked_alpha[label_index])
                openmax_score += [modified_score]
                openmax_score_unknown += [activation_vector[label_index] - modified_score]
            elif score_cal_type == 3:
                modified_score = activation_vector[label_index] * (
                        1 - ((wscore + model_wscore) / 2) * ranked_alpha[label_index])
                openmax_score += [modified_score]
                openmax_score_unknown += [activation_vector[label_index] - modified_score]

            if savepath != '':
                ret_str += str(modified_score) + ','
                ret_str += str(activation_vector[label_index] - modified_score) + ','

    openmax_score = np.array(openmax_score)
    openmax_score_unknown = np.array(openmax_score_unknown)

    # Pass the re-calibrated scores for the image into OpenMax
    openmax_probab = compute_openmax_probability(openmax_score, openmax_score_unknown, labels)
    softmax_probab = compute_softmax_probability(activation_vector)  # Calculate SoftMax ???

    ret_openmax_probab = []
    rest_rate = 0

    if modify == 1:
        for i in range(len(openmax_probab)):
            el = openmax_probab[i]
            if savepath != '':
                ret_str += str(el) + ','
            if i < (len(openmax_probab) - 1):
                modified_score_final = el * (1 - openmax_wscoe[i])
                if modified_score_final < 0.00001:
                    modified_score_final = 0.00001
                ret_openmax_probab.append(modified_score_final)

                gap_tmp_final = el - modified_score_final

                if gap_tmp_final >= 0:
                    rest_rate += gap_tmp_final
                else:
                    rest_rate += 0
            else:
                ret_openmax_probab.append(el + rest_rate)

        if savepath != '':
            for el in ret_openmax_probab:
                ret_str += str(el) + ','

            for i in range(softmax_probab.size):
                ret_str += str(softmax_probab[i]) + ','

            with open(savepath, 'a') as f:
                f.write(str(ad) + ',' + str(cn) + ',' + ret_str + '\n')

        return np.array(ret_openmax_probab), softmax_probab
    elif modify == 0:
        if savepath != '':
            for i in range(openmax_probab.size):
                ret_str += str(openmax_probab[i]) + ','

            for i in range(softmax_probab.size):
                ret_str += str(softmax_probab[i]) + ','

            with open(savepath, 'a') as f:
                f.write(str(ad) + ',' + str(cn) + ',' + ret_str + '\n')

        return np.array(openmax_probab), softmax_probab

# 利用 sc 和 m03 的数据补全 bl 阶段的数据
def combine_m03_and_sc_into_bl(df, isImage=False, Modality='MRI'):
    if ('VISCODE2' not in df.columns) and ('VISCODE' in df.columns):
        df.rename(columns={'VISCODE': 'VISCODE2'}, inplace=True)  # 为了统一表达 VISCODE 修改为VISCODE2

    rID_set = df['RID'].drop_duplicates()
    # df = df.set_index(['RID', 'VISCODE2'], drop=False)
    for index, row in rID_set.items():
        rRID = row
        if isImage:
            sub_df = df.loc[(df['RID'] == rRID) & (df['Modality'] == Modality) & (
                    (df['VISCODE2'] == 'm03') | (df['VISCODE2'] == 'bl') | (df['VISCODE2'] == 'sc') | (
                    df['VISCODE2'] == 'scmri'))]
        else:
            sub_df = df.loc[(df['RID'] == rRID) & (
                    (df['VISCODE2'] == 'm03') | (df['VISCODE2'] == 'bl') | (df['VISCODE2'] == 'sc') | (
                    df['VISCODE2'] == 'scmri'))]

        # if (row == '4858' and isImage and Modality == 'MRI'):
        #    print('%%%%%%%%%%%%%%%%%%%%%%%%%')
        #    print(sub_df)
        # print('xxxxxxxxxxxxxxxxxx',row)
        # print('子模块图像数量：',sub_df.shape[0])
        if sub_df.shape[0] > 0:
            bl_flag = sub_df['VISCODE2'].isin(['bl']).any()
            sc_flag = sub_df['VISCODE2'].isin(['sc', 'scmri']).any()
            m03_flage = sub_df['VISCODE2'].isin(['m03']).any()
            ##bl_flag=sub_df['RID'].isin(['bl']).any()
            ##sc_flag=sub_df['RID'].isin(['sc','scmri']).any()
            ##m03_flage=sub_df['RID'].isin(['m03']).any()

            if bl_flag and (sc_flag or m03_flage):  # bl 与其他两个阶段中的一个或两个 共存
                columns = sub_df.columns.values.tolist()
                mark = []
                for index_order, row in sub_df.iterrows():
                    visit = row['VISCODE2']
                    if visit == 'bl':
                        for i in range(0, len(columns)):
                            tmp_value = row[columns[i]]
                            if pd.isnull(tmp_value) or tmp_value == '-4' or tmp_value == '\"\"' or tmp_value == '':
                                mark.append([index_order, columns[i]])

                if not isImage:  # 不修补图像的基本信息
                    for i in range(0, len(mark)):
                        index = mark[i][0]
                        columns_name = mark[i][1]
                        for index_order, row in sub_df.iterrows():
                            visit = row['VISCODE2']
                            if visit != 'bl':
                                other_value = row[columns_name]
                                if not (pd.isnull(
                                        other_value) or other_value == '-4' or other_value == '\"\"' or other_value == ''):
                                    # print('修改前：',df.at[index, columns_name])
                                    df.at[index, columns_name] = other_value
                                    # print('修改后：',df.at[index, columns_name])
                                    break
            elif (not bl_flag) and (sc_flag and m03_flage):  # bl 不存在，sc和m03都存在
                columns = sub_df.columns.values.tolist()
                mark = []
                change_visitcoede = []
                for index_order, row in sub_df.iterrows():
                    visit = row['VISCODE2']
                    if visit == 'sc' or visit == 'scmri':
                        change_visitcoede.append(index_order)
                        for i in range(0, len(columns)):
                            tmp_value = row[columns[i]]
                            if pd.isnull(tmp_value) or tmp_value == '-4' or tmp_value == '\"\"' or tmp_value == '':
                                mark.append([index_order, columns[i]])

                if not isImage:  # 不修补图像的基本信息
                    for i in range(0, len(mark)):
                        index = mark[i][0]
                        columns_name = mark[i][1]
                        for index_order, row in sub_df.iterrows():
                            visit = row['VISCODE2']
                            if not (visit == 'sc' or visit == 'scmri'):
                                other_value = row[columns_name]
                                if not (pd.isnull(
                                        other_value) or other_value == '-4' or other_value == '\"\"' or other_value == ''):
                                    # print('修改前：',df.at[index, columns_name])
                                    df.at[index, columns_name] = other_value
                                    # print('修改后：',df.at[index, columns_name])
                                    break

                for index in change_visitcoede:
                    # print('修改前：',df.at[index_order, 'VISCODE2'])
                    df.at[index, 'VISCODE2'] = 'bl'
                    # print('修改后：',df.at[index_order, 'VISCODE2'])
            elif (not (bl_flag or m03_flage)) and sc_flag:  # 只有 sc 阶段存在
                for index_order, row in sub_df.iterrows():
                    # print('修改前：',df.at[index_order, 'VISCODE2'])
                    df.at[index_order, 'VISCODE2'] = 'bl'
                    # print('修改后：',df.at[index_order, 'VISCODE2'])
            elif (not (bl_flag or sc_flag)) and m03_flage:  # 只有m03阶段存在
                for index_order, row in sub_df.iterrows():
                    # print('修改前：',df.at[index_order, 'VISCODE2'])
                    df.at[index_order, 'VISCODE2'] = 'bl'
                    # print('修改后：',df.at[index_order, 'VISCODE2'])
        # print('xxxxxxxxxxxxxxxxxx',row)
        # print('xxxxxxxxxxxxxxxxxx',isImage)
        # print('xxxxxxxxxxxxxxxxxx',Modality)
        # if ((rRID == '4858') and isImage and (Modality == 'MRI')):
        #    print('%%%%%%%%%%%%%%%%%%%%%%%%%')
        #    sdf = df.loc[(df['RID'] == rRID) & (df['Modality'] == Modality)]
        #    print(sdf)
    return df


# df 缺失值填充
def set_missing_value(df):
    # df.where(df.notnull(),'-4')
    df = df.fillna('-4')
    df = df.where(df != '', '-4')
    df = df.where(df != '\"\"', '-4')
    return df


scale_path = '/data/huangyunyou/ADNI/SCALE/'
scale_base = scale_path + 'base_scale.csv'

# 量表数据
base_df = pd.read_csv(scale_base, dtype=str)
base_df = combine_m03_and_sc_into_bl(base_df)
base_df = set_missing_value(base_df)

base_df_mh = base_df[['RID', 'VISCODE2', 'MHPSYCH', 'MH2NEURL', 'MH2NEURL_CON']].drop_duplicates(
    subset=['RID', 'VISCODE2'])

base_df_mh = base_df_mh.set_index(['RID', 'VISCODE2'], drop=False)

base_df_sy = base_df[
    ['RID', 'VISCODE2', 'AXNAUSEA', 'AXVOMIT', 'AXDIARRH', 'AXCONSTP', 'AXABDOMN', 'AXSWEATN', 'AXDIZZY',
     'AXENERGY', 'AXDROWSY', 'AXVISION', 'AXHDACHE', 'AXDRYMTH', 'AXBREATH', 'AXCOUGH', 'AXPALPIT', 'AXCHEST',
     'AXURNDIS', 'AXURNFRQ', 'AXANKLE', 'AXMUSCLE', 'AXRASH', 'AXINSOMN', 'AXDPMOOD', 'AXCRYING', 'AXELMOOD',
     'AXWANDER', 'AXFALL', 'AXOTHER']].drop_duplicates(subset=['RID', 'VISCODE2'])

base_df_sy = base_df_sy.set_index(['RID', 'VISCODE2'], drop=False)

merger_path = '/data/huangyunyou/ADNI/ADNIMERGE_without_bad_value.csv'
merge_df = pd.read_csv(merger_path, dtype=str)  # 综合信息
# merge_df.rename(columns={'VISCODE': 'VISCODE2'}, inplace=True)  # 为了统一表达 VISCODE 修改为VISCODE2
merge_df = combine_m03_and_sc_into_bl(merge_df)
merge_df = set_missing_value(merge_df)
merge_df = merge_df.set_index(['RID', 'VISCODE2'], drop=False)

scale_cog = scale_path + 'cog_scale.csv'
cog_df = pd.read_csv(scale_cog, dtype=str)
cog_df = combine_m03_and_sc_into_bl(cog_df)
cog_df = set_missing_value(cog_df).drop_duplicates(subset=['RID', 'VISCODE2'])
cog_df = cog_df.set_index(['RID', 'VISCODE2'], drop=False)


def get_core_evidence_4v(rids, viscodes, batch, training=True):
    ret_value = []
    for i in range(batch):
        if training:
            rid = rids[i]
            viscode = viscodes[i]
        else:
            rid = str(rids[i], encoding="utf-8")
            viscode = str(viscodes[i], encoding="utf-8")

        # tmp_sample=[]
        cong = merge_df.loc[(rid, viscode), ['CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'MOCA', 'mPACCdigit',
                                             'mPACCtrailsB']].to_list()

        if (rid, viscode) in base_df_mh.index:
            based_array = base_df_mh.loc[(rid, viscode), ['MHPSYCH', 'MH2NEURL', 'MH2NEURL_CON']].to_list()
        else:
            based_array = ['-4', '-4', '-4']

        if based_array[1] == '-4' and based_array[2] != '-4':
            based_array[1] = based_array[2]
        based_array = based_array[0:2]

        if (rid, viscode) in base_df_sy.index:
            symptom_array = base_df_sy.loc[
                (rid, viscode), ['AXNAUSEA', 'AXVOMIT', 'AXDIARRH', 'AXCONSTP', 'AXABDOMN', 'AXSWEATN', 'AXDIZZY',
                                 'AXENERGY', 'AXDROWSY', 'AXVISION', 'AXHDACHE', 'AXDRYMTH', 'AXBREATH', 'AXCOUGH',
                                 'AXPALPIT', 'AXCHEST', 'AXURNDIS', 'AXURNFRQ', 'AXANKLE', 'AXMUSCLE', 'AXRASH',
                                 'AXINSOMN', 'AXDPMOOD', 'AXCRYING', 'AXELMOOD', 'AXWANDER', 'AXFALL',
                                 'AXOTHER']].to_list()
        else:
            symptom_array = ['-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4',
                             '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4']
        for j in range(len(symptom_array)):
            if float(symptom_array[j]) == -1.0:
                symptom_array[j] = '-4'

        sum_abnormal_array = ['-4', '-4']

        if '-4' not in symptom_array:
            sum_abnormal = 0
            for j in range(len(symptom_array)):
                el = symptom_array[j]
                if int(el) == 2:
                    sum_abnormal += 1
                if j == 20:
                    sum_abnormal_array[0] = sum_abnormal
            sum_abnormal_array[1] = sum_abnormal

        if (rid, viscode) in cog_df.index:
            cci = cog_df.loc[
                (rid, viscode), ['CCI1', 'CCI2', 'CCI3', 'CCI4', 'CCI5', 'CCI6', 'CCI7', 'CCI8', 'CCI9', 'CCI10',
                                 'CCI11', 'CCI12', 'CCI13', 'CCI14', 'CCI15', 'CCI16', 'CCI17', 'CCI18', 'CCI19',
                                 'CCI20', ]].to_list()
        else:
            cci = ['-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4',
                   '-4', '-4', '-4']

        cci_array = ['-4', '-4']
        if '-4' not in cci:
            sum_value = 0
            for j in range(len(cci)):
                sum_value += float(cci[j])
                if j == 11:
                    cci_array[0] = sum_value
            cci_array[1] = sum_value

        cong.extend(cci_array)
        cong.extend(sum_abnormal_array)
        cong.extend(based_array)
        ret_value.append(cong)
    return np.array(ret_value)


def clear_evidence_2v(rids, viscodes, evidences, ads, cns, mcis):
    ret_value = []
    ret_rid_value = []
    ret_ads_value = []
    ret_cn_value = []
    ret_mci_value = []
    ret_viscoed_value = []
    rid_viscode_set = set()
    for i in range(rids.size):
        rid = rids[i]
        viscode = viscodes[i]
        key = str(rid) + '_' + str(viscode)
        # print('************************* ', key)
        if key not in rid_viscode_set:
            ret_value.append(evidences[i])
            ret_rid_value.append(rids[i])
            ret_ads_value.append(ads[i])
            ret_cn_value.append(cns[i])
            ret_mci_value.append(mcis[i])
            ret_viscoed_value.append(viscodes[i])
            rid_viscode_set.add(key)
        # else:
        #    print('not not not !   ',i)
    return np.array(ret_value), np.array(ret_rid_value), np.array(ret_ads_value), np.array(ret_cn_value), np.array(
        ret_mci_value), np.array(ret_viscoed_value)


def get_core_evidence_convert_abnormal(rids, ad, evidences, evidence_save_path='', isTraining=True, data_len=7,
                                       low_percent=5, high_percent=95):
    # print(evidences)
    sample_data = {}
    max_array = np.full(data_len, -5000000)
    min_array = np.full(data_len, 5000000)

    ad_num = np.zeros(data_len)
    cn_num = np.zeros(data_len)

    ad_sum = np.zeros(data_len)
    cn_sum = np.zeros(data_len)

    for i in range(ad.size):
        lable = int(ad[i])
        rid = rids[i]
        tmp_data = evidences[i]
        for j in range(tmp_data.size):
            c_value = tmp_data[j]
            if c_value != '-4' and c_value != '-4.0' and c_value != '\"\"' and c_value != '':
                c_value = float(c_value)

                if isTraining:
                    if c_value > max_array[j]:
                        max_array[j] = c_value
                    if c_value < min_array[j]:
                        min_array[j] = c_value

                    if lable == 1:
                        ad_sum[j] += c_value
                        ad_num[j] += 1
                    else:
                        cn_sum[j] += c_value
                        cn_num[j] += 1

                if rid not in sample_data:
                    tmp_array = []
                    tmp_array.append(np.zeros(data_len))
                    tmp_array.append(np.zeros(data_len))
                    tmp_array.append(np.zeros(data_len))
                    tmp_array.append(np.zeros(data_len))
                    sample_data[rid] = tmp_array

                tmp_rid_sample = sample_data[rid]

                if lable == 1:
                    tmp_rid_sample[0][j] += c_value
                    tmp_rid_sample[1][j] += 1
                else:
                    tmp_rid_sample[2][j] += c_value
                    tmp_rid_sample[3][j] += 1

    if not isTraining:

        max_array = np.load(evidence_save_path + 'max.npy')
        min_array = np.load(evidence_save_path + 'min.npy')

        ad_num = np.load(evidence_save_path + 'ad_num.npy')
        cn_num = np.load(evidence_save_path + 'cn_num.npy')

        ad_sum = np.load(evidence_save_path + 'ad_sum.npy')
        cn_sum = np.load(evidence_save_path + 'cn_sum.npy')
    else:

        if os.path.exists(evidence_save_path + 'max.npy'):
            os.remove(evidence_save_path + 'max.npy')

        if os.path.exists(evidence_save_path + 'min.npy'):
            os.remove(evidence_save_path + 'min.npy')

        if os.path.exists(evidence_save_path + 'ad_sum.npy'):
            os.remove(evidence_save_path + 'ad_sum.npy')

        if os.path.exists(evidence_save_path + 'ad_num.npy'):
            os.remove(evidence_save_path + 'ad_num.npy')

        if os.path.exists(evidence_save_path + 'cn_sum.npy'):
            os.remove(evidence_save_path + 'cn_sum.npy')

        if os.path.exists(evidence_save_path + 'cn_num.npy'):
            os.remove(evidence_save_path + 'cn_num.npy')

        np.save(evidence_save_path + 'max.npy', max_array)
        np.save(evidence_save_path + 'min.npy', min_array)
        np.save(evidence_save_path + 'ad_sum.npy', ad_sum)
        np.save(evidence_save_path + 'ad_num.npy', ad_num)
        np.save(evidence_save_path + 'cn_sum.npy', cn_sum)
        np.save(evidence_save_path + 'cn_num.npy', cn_num)

    gaps = max_array - min_array
    means_ad = ad_sum / ad_num
    means_cn = cn_sum / cn_num

    for i in range(len(ad_num)):
        if ad_num[i] < 1:
            means_ad[i] = 1

    for i in range(len(cn_num)):
        if cn_num[i] < 1:
            means_cn[i] = 1

    print(max_array)
    print(min_array)
    print(means_ad)
    print(means_cn)

    data_ad = []
    data_cn = []
    for i in range(ad.size):
        lable = int(ad[i])
        rid = rids[i]
        tmp_data = evidences[i]
        for j in range(tmp_data.size):
            min_value = min_array[j]
            gap = gaps[j]
            c_value = tmp_data[j]
            if c_value != '-4' and c_value != '-4.0' and c_value != '\"\"' and c_value != '':
                # c_value = float(c_value)
                evidences[i][j] = c_value
            else:
                rid_c_value = 0
                if rid in sample_data:
                    tmp_rid_sample = sample_data[rid]
                    if lable == 1:
                        rid_sum = tmp_rid_sample[0][j]
                        rid_num = tmp_rid_sample[1][j]
                        if j == (tmp_data.size - 2) or j == (tmp_data.size - 1):
                            if rid_num > 0:
                                rid_c_value = round(rid_sum / rid_num, 0)
                            else:
                                rid_c_value = round(means_ad[j], 0)
                        else:
                            if rid_num > 0:
                                rid_c_value = rid_sum / rid_num
                            else:
                                rid_c_value = means_ad[j]
                    else:
                        rid_sum = tmp_rid_sample[2][j]
                        rid_num = tmp_rid_sample[3][j]
                        if j == (tmp_data.size - 2) or j == (tmp_data.size - 1):
                            if rid_num > 0:
                                rid_c_value = round(rid_sum / rid_num, 0)
                            else:
                                rid_c_value = round(means_cn[j], 0)
                        else:
                            if rid_num > 0:
                                rid_c_value = rid_sum / rid_num
                            else:
                                rid_c_value = means_cn[j]
                else:
                    if j == (tmp_data.size - 2) or j == (tmp_data.size - 1):
                        if lable == 1:
                            rid_c_value = round(means_ad[j], 0)
                        else:
                            rid_c_value = round(means_cn[j], 0)
                    else:
                        if lable == 1:
                            rid_c_value = means_ad[j]
                        else:
                            rid_c_value = means_cn[j]

                evidences[i][j] = rid_c_value
        if lable == 1:
            data_ad.append(evidences[i])
        elif lable == 0:
            data_cn.append(evidences[i])

    evidences = evidences.astype(np.float32)  # 填补空缺值之后的数据集

    data_ad = np.array(data_ad).astype(np.float32)
    data_cn = np.array(data_cn).astype(np.float32)

    evidences_1 = np.zeros(evidences.shape)  # 归一化的数据集
    evidences_ad_abnor = np.zeros(evidences.shape)  # 与ad对比的异常
    evidences_cn_abnor = np.zeros(evidences.shape)  # 与正常人对比的异常

    ad_evidences_thr = np.zeros(2 * data_len)
    cn_evidences_thr = np.zeros(2 * data_len)

    if isTraining:
        # region 统计正常值
        # cdr
        ad_evidences_thr[0] = np.percentile(data_ad[:, 0], low_percent)
        ad_evidences_thr[1] = data_ad[:, 0].max()
        cn_evidences_thr[0] = 0
        cn_evidences_thr[1] = 0

        print('cdr AD low: ', ad_evidences_thr[0])
        print('cdr AD high: ', ad_evidences_thr[1])
        print('cdr CN low: ', cn_evidences_thr[0])
        print('cdr CN high:', cn_evidences_thr[1])

        # adas11
        ad_evidences_thr[2] = np.percentile(data_ad[:, 1], low_percent)
        ad_evidences_thr[3] = data_ad[:, 1].max()
        cn_evidences_thr[2] = 0
        cn_evidences_thr[3] = np.percentile(data_cn[:, 1], high_percent)

        print('adas11 AD low: ', ad_evidences_thr[2])
        print('adas11 AD high: ', ad_evidences_thr[3])
        print('adas11 CN low: ', cn_evidences_thr[2])
        print('adas11 CN high:', cn_evidences_thr[3])

        # adas13
        ad_evidences_thr[4] = np.percentile(data_ad[:, 2], low_percent)
        ad_evidences_thr[5] = data_ad[:, 2].max()
        cn_evidences_thr[4] = 0
        cn_evidences_thr[5] = np.percentile(data_cn[:, 2], high_percent)
        # print('adas13 cn high : ', cn_evidences_thr[5])

        print('adas13 AD low: ', ad_evidences_thr[4])
        print('adas13 AD high: ', ad_evidences_thr[5])
        print('adas13 CN low: ', cn_evidences_thr[4])
        print('adas13 CN high:', cn_evidences_thr[5])

        # adasQ4
        ad_evidences_thr[6] = np.percentile(data_ad[:, 3], low_percent)
        ad_evidences_thr[7] = data_ad[:, 3].max()
        cn_evidences_thr[6] = 0
        cn_evidences_thr[7] = np.percentile(data_cn[:, 3], high_percent)
        # print('adasQ4 cn high : ', cn_evidences_thr[7])

        print('adasQ4 AD low: ', ad_evidences_thr[6])
        print('adasQ4 AD high: ', ad_evidences_thr[7])
        print('adasQ4 CN low: ', cn_evidences_thr[6])
        print('adasQ4 CN high:', cn_evidences_thr[7])

        # nmse
        ad_evidences_thr[8] = 0
        ad_evidences_thr[9] = np.percentile(data_ad[:, 4], high_percent)
        cn_evidences_thr[8] = 25
        cn_evidences_thr[9] = data_cn[:, 4].max()

        print('nmse AD low: ', ad_evidences_thr[8])
        print('nmse AD high: ', ad_evidences_thr[9])
        print('nmse CN hilowgh: ', cn_evidences_thr[8])
        print('nmse CN high:', cn_evidences_thr[9])

        # moca
        ad_evidences_thr[10] = 0
        ad_evidences_thr[11] = np.percentile(data_ad[:, 5], high_percent)
        cn_evidences_thr[10] = 26
        cn_evidences_thr[11] = data_cn[:, 5].max()
        # print('moca cn low : ', cn_evidences_thr[10])

        print('moca AD low: ', ad_evidences_thr[10])
        print('moca AD high: ', ad_evidences_thr[11])
        print('moca CN low: ', cn_evidences_thr[10])
        print('moca CN high:', cn_evidences_thr[11])

        # mpacc
        ad_evidences_thr[12] = np.percentile(data_ad[:, 6], low_percent)
        ad_evidences_thr[13] = np.percentile(data_ad[:, 6], high_percent)
        cn_evidences_thr[12] = np.percentile(data_cn[:, 6], low_percent)
        cn_evidences_thr[13] = np.percentile(data_cn[:, 6], high_percent)

        print('mpacc AD low: ', ad_evidences_thr[12])
        print('mpacc AD high: ', ad_evidences_thr[13])
        print('mpacc CN low: ', cn_evidences_thr[12])
        print('mpacc CN high:', cn_evidences_thr[13])

        # mpaccb
        ad_evidences_thr[14] = np.percentile(data_ad[:, 7], low_percent)
        ad_evidences_thr[15] = np.percentile(data_ad[:, 7], high_percent)
        cn_evidences_thr[14] = np.percentile(data_cn[:, 7], low_percent)
        cn_evidences_thr[15] = np.percentile(data_cn[:, 7], high_percent)

        print('mpaccb AD low: ', ad_evidences_thr[14])
        print('mpaccb AD high: ', ad_evidences_thr[15])
        print('mpaccb CN low: ', cn_evidences_thr[14])
        print('mpaccb CN high:', cn_evidences_thr[15])

        # cci12
        ad_evidences_thr[16] = np.percentile(data_ad[:, 8], low_percent)
        ad_evidences_thr[17] = 60
        cn_evidences_thr[16] = 12
        cn_evidences_thr[17] = np.percentile(data_cn[:, 8], high_percent)

        print('cci12 AD low: ', ad_evidences_thr[16])
        print('cci12 AD high: ', ad_evidences_thr[17])
        print('cci12 CN low: ', cn_evidences_thr[16])
        print('cci12 CN high:', cn_evidences_thr[17])

        # cci20
        ad_evidences_thr[18] = np.percentile(data_ad[:, 9], low_percent)
        ad_evidences_thr[19] = 100
        cn_evidences_thr[18] = 20
        cn_evidences_thr[19] = np.percentile(data_cn[:, 9], high_percent)

        print('cci20 AD low: ', ad_evidences_thr[18])
        print('cci20 AD high: ', ad_evidences_thr[19])
        print('cci20 CN low: ', cn_evidences_thr[18])
        print('cci20 CN high:', cn_evidences_thr[19])

        # AX21
        ad_evidences_thr[20] = np.percentile(data_ad[:, 10], low_percent)
        ad_evidences_thr[21] = np.percentile(data_ad[:, 10], high_percent)
        cn_evidences_thr[20] = np.percentile(data_cn[:, 10], low_percent)
        cn_evidences_thr[21] = np.percentile(data_cn[:, 10], high_percent)

        print('AX21 AD low: ', ad_evidences_thr[20])
        print('AX21 AD high: ', ad_evidences_thr[21])
        print('AX21 CN low: ', cn_evidences_thr[20])
        print('AX21 CN high:', cn_evidences_thr[21])

        # AX28
        ad_evidences_thr[22] = np.percentile(data_ad[:, 11], low_percent)
        ad_evidences_thr[23] = np.percentile(data_ad[:, 11], high_percent)
        cn_evidences_thr[22] = np.percentile(data_cn[:, 11], low_percent)
        cn_evidences_thr[23] = np.percentile(data_cn[:, 11], high_percent)

        print('AX28 AD low: ', ad_evidences_thr[22])
        print('AX28 AD high: ', ad_evidences_thr[23])
        print('AX28 CN low: ', cn_evidences_thr[22])
        print('AX28 CN high:', cn_evidences_thr[23])

        # MHPSYCH
        ad_evidences_thr[24] = Counter(data_ad[:, 12]).most_common(1)[0][0]
        ad_evidences_thr[25] = ad_evidences_thr[24] - 100
        cn_evidences_thr[24] = Counter(data_cn[:, 12]).most_common(1)[0][0]
        cn_evidences_thr[25] = cn_evidences_thr[24] - 100

        print('MHPSYCH AD low: ', ad_evidences_thr[24])
        print('MHPSYCH AD high: ', ad_evidences_thr[25])
        print('MHPSYCH CN low: ', cn_evidences_thr[24])
        print('MHPSYCH CN high:', cn_evidences_thr[25])

        # MH2NEURL
        ad_evidences_thr[26] = Counter(data_ad[:, 13]).most_common(1)[0][0]
        ad_evidences_thr[27] = ad_evidences_thr[26] - 100
        cn_evidences_thr[26] = Counter(data_cn[:, 13]).most_common(1)[0][0]
        cn_evidences_thr[27] = cn_evidences_thr[26] - 100

        print('MH2NEURL AD low: ', ad_evidences_thr[26])
        print('MH2NEURL AD high: ', ad_evidences_thr[27])
        print('MH2NEURL CN low: ', cn_evidences_thr[26])
        print('MH2NEURL CN high:', cn_evidences_thr[27])
        # endregion
        if os.path.exists(evidence_save_path + 'ad_thr.npy'):
            os.remove(evidence_save_path + 'ad_thr.npy')

        if os.path.exists(evidence_save_path + 'cn_thr.npy'):
            os.remove(evidence_save_path + 'cn_thr.npy')

        np.save(evidence_save_path + 'ad_thr.npy', ad_evidences_thr)
        np.save(evidence_save_path + 'cn_thr.npy', cn_evidences_thr)
    else:
        ad_evidences_thr = np.load(evidence_save_path + 'ad_thr.npy')
        cn_evidences_thr = np.load(evidence_save_path + 'cn_thr.npy')

    evidence_nor = []

    for i in range(ad.size):
        tmp_sample = evidences[i]
        tmp_nor_distan = [0, 0]

        for j in range(tmp_sample.size):
            c_value = tmp_sample[j]
            evidences_1[i][j] = (c_value - min_array[j]) / gaps[j]
            ad_nor_low = ad_evidences_thr[2 * j]
            ad_nor_high = ad_evidences_thr[2 * j + 1]

            cn_nor_low = cn_evidences_thr[2 * j]
            cn_nor_high = cn_evidences_thr[2 * j + 1]

            if ad_nor_high < ad_nor_low:
                if c_value != ad_nor_low:
                    tmp_nor_distan[0] += 1
                    # evidences_2[i][j]=1
                    evidences_ad_abnor[i][j] = 1
            else:
                if not (c_value >= ad_nor_low and c_value <= ad_nor_high):
                    if c_value < ad_nor_low:
                        if ad_nor_low == 0:
                            tmp_nor_distan[0] += 1
                            # evidences_2
                            evidences_ad_abnor[i][j] = 1
                        else:
                            tmp_nor_distan[0] += (ad_nor_low - c_value + abs(ad_nor_low)) / abs(ad_nor_low)
                            evidences_ad_abnor[i][j] = (ad_nor_low - c_value + abs(ad_nor_low)) / abs(ad_nor_low)
                    if c_value > ad_nor_high:
                        if ad_nor_high == 0:
                            tmp_nor_distan[0] += 1
                            evidences_ad_abnor[i][j] = 1
                        else:
                            tmp_nor_distan[0] += (c_value - ad_nor_high + abs(ad_nor_high)) / abs(ad_nor_high)
                            evidences_ad_abnor[i][j] = (c_value - ad_nor_high + abs(ad_nor_high)) / abs(ad_nor_high)

            if cn_nor_high < cn_nor_low:
                if c_value != cn_nor_low:
                    tmp_nor_distan[1] += 1
                    evidences_cn_abnor[i][j] = 1
            else:
                if not (c_value >= cn_nor_low and c_value <= cn_nor_high):
                    if c_value < cn_nor_low:
                        if cn_nor_low == 0:
                            tmp_nor_distan[1] += 1
                            evidences_cn_abnor[i][j] = 1
                        else:
                            tmp_nor_distan[1] += (cn_nor_low - c_value + abs(cn_nor_low)) / abs(cn_nor_low)
                            evidences_cn_abnor[i][j] = (cn_nor_low - c_value + abs(cn_nor_low)) / abs(cn_nor_low)
                    if c_value > cn_nor_high:
                        if cn_nor_high == 0:
                            tmp_nor_distan[1] += 1
                            evidences_cn_abnor[i][j] = 1
                        else:
                            tmp_nor_distan[1] += (c_value - cn_nor_high + abs(cn_nor_high)) / abs(cn_nor_high)
                            evidences_cn_abnor[i][j] = (c_value - cn_nor_high + abs(cn_nor_high)) / abs(cn_nor_high)

        evidence_nor.append(tmp_nor_distan)
    return np.array(evidence_nor), evidences_1, np.concatenate((evidences_ad_abnor, evidences_cn_abnor), axis=1)

def get_core_evidence_convert_abnormal_2v_adcn_mcismc(rids, ad, cn, mci, evidences, evidence_save_path='',
                                                      isTraining=False, data_len=7):
    sample_data = {}

    mci_num = np.zeros(data_len)
    smc_num = np.zeros(data_len)

    mci_sum = np.zeros(data_len)
    smc_sum = np.zeros(data_len)

    for i in range(ad.size):
        lable = int(ad[i])
        lable_mci = int(mci[i])
        lable_cn = int(cn[i])
        rid = rids[i]
        tmp_data = evidences[i]
        for j in range(tmp_data.size):
            c_value = tmp_data[j]
            if c_value != '-4' and c_value != '-4.0' and c_value != '\"\"' and c_value != '':
                c_value = float(c_value)

                if lable_mci == 1:
                    mci_sum[j] += c_value
                    mci_num[j] += 1

                if lable == 0 and lable_cn == 0 and lable_mci == 0:
                    smc_sum[j] += c_value
                    smc_num[j] += 1

                if rid not in sample_data:
                    tmp_array = []
                    tmp_array.append(np.zeros(data_len))
                    tmp_array.append(np.zeros(data_len))
                    tmp_array.append(np.zeros(data_len))
                    tmp_array.append(np.zeros(data_len))
                    tmp_array.append(np.zeros(data_len))
                    tmp_array.append(np.zeros(data_len))
                    tmp_array.append(np.zeros(data_len))
                    tmp_array.append(np.zeros(data_len))
                    sample_data[rid] = tmp_array

                tmp_rid_sample = sample_data[rid]

                if lable == 1:
                    tmp_rid_sample[0][j] += c_value
                    tmp_rid_sample[1][j] += 1
                elif lable_cn == 1:
                    tmp_rid_sample[2][j] += c_value
                    tmp_rid_sample[3][j] += 1
                elif lable_mci == 1:
                    tmp_rid_sample[4][j] += c_value
                    tmp_rid_sample[5][j] += 1
                elif lable == 0 and lable_cn == 0 and lable_mci == 0:
                    tmp_rid_sample[6][j] += c_value
                    tmp_rid_sample[7][j] += 1

    max_array = np.load(evidence_save_path + 'max.npy')
    min_array = np.load(evidence_save_path + 'min.npy')

    ad_num = np.load(evidence_save_path + 'ad_num.npy')
    cn_num = np.load(evidence_save_path + 'cn_num.npy')

    ad_sum = np.load(evidence_save_path + 'ad_sum.npy')
    cn_sum = np.load(evidence_save_path + 'cn_sum.npy')

    gaps = max_array - min_array
    means_ad = ad_sum / ad_num
    means_cn = cn_sum / cn_num
    means_mci = mci_sum / mci_num
    means_smc = smc_sum / smc_num

    for i in range(len(ad_num)):
        if ad_num[i] < 1:
            means_ad[i] = 1

    for i in range(len(cn_num)):
        if cn_num[i] < 1:
            means_cn[i] = 1

    for i in range(len(mci_num)):
        if mci_num[i] < 1:
            means_mci[i] = means_cn[i] * 1.3

    for i in range(len(smc_num)):
        if smc_num[i] < 1:
            means_smc[i] = means_cn[i] * 1.1

    print(max_array)
    print(min_array)
    print(means_ad)
    print(means_cn)
    print(means_mci)
    print(means_smc)

    for i in range(ad.size):
        lable = int(ad[i])
        lable_cn = int(cn[i])
        rid = rids[i]
        lable_mci = int(mci[i])
        tmp_data = evidences[i]
        for j in range(tmp_data.size):
            min_value = min_array[j]
            gap = gaps[j]
            c_value = tmp_data[j]
            if c_value != '-4' and c_value != '-4.0' and c_value != '\"\"' and c_value != '':
                c_value = float(c_value)
                evidences[i][j] = c_value
            else:
                rid_c_value = 0
                if rid in sample_data:
                    tmp_rid_sample = sample_data[rid]
                    if lable == 1:
                        rid_sum = tmp_rid_sample[0][j]
                        rid_num = tmp_rid_sample[1][j]
                        if j == (tmp_data.size - 2) or j == (tmp_data.size - 1):
                            if rid_num > 0:
                                rid_c_value = round(rid_sum / rid_num)
                            else:
                                rid_c_value = round(means_ad[j])
                        else:
                            if rid_num > 0:
                                rid_c_value = rid_sum / rid_num
                            else:
                                rid_c_value = means_ad[j]

                    elif lable_cn == 1:
                        rid_sum = tmp_rid_sample[2][j]
                        rid_num = tmp_rid_sample[3][j]
                        if j == (tmp_data.size - 2) or j == (tmp_data.size - 1):
                            if rid_num > 0:
                                rid_c_value = round(rid_sum / rid_num, 0)
                            else:
                                rid_c_value = round(means_cn[j], 0)
                        else:
                            if rid_num > 0:
                                rid_c_value = rid_sum / rid_num
                            else:
                                rid_c_value = means_cn[j]

                    elif lable_mci == 1:
                        rid_sum = tmp_rid_sample[4][j]
                        rid_num = tmp_rid_sample[5][j]
                        if j == (tmp_data.size - 2) or j == (tmp_data.size - 1):
                            if rid_num > 0:
                                rid_c_value = round(rid_sum / rid_num, 0)
                            else:
                                rid_c_value = round(means_mci[j], 0)
                        else:
                            if rid_num > 0:
                                rid_c_value = rid_sum / rid_num
                            else:
                                rid_c_value = means_mci[j]
                    elif lable == 0 and lable_cn == 0 and lable_mci == 0:
                        rid_sum = tmp_rid_sample[6][j]
                        rid_num = tmp_rid_sample[7][j]
                        if j == (tmp_data.size - 2) or j == (tmp_data.size - 1):
                            if rid_num > 0:
                                rid_c_value = round(rid_sum / rid_num, 0)
                            else:
                                rid_c_value = round(means_smc[j], 0)
                        else:
                            if rid_num > 0:
                                rid_c_value = rid_sum / rid_num
                            else:
                                rid_c_value = means_smc[j]

                else:
                    if j == (tmp_data.size - 2) or j == (tmp_data.size - 1):
                        if lable == 1:
                            rid_c_value = round(means_ad[j], 0)
                        elif lable_cn == 1:
                            rid_c_value = round(means_cn[j], 0)
                        elif lable_mci == 1:
                            rid_c_value = round(means_mci[j], 0)
                        elif lable == 0 and lable_cn == 0 and lable_mci == 0:
                            rid_c_value = round(means_smc[j], 0)
                    else:
                        if lable == 1:
                            rid_c_value = means_ad[j]
                        elif lable_cn == 1:
                            rid_c_value = means_cn[j]
                        elif lable_mci == 1:
                            rid_c_value = means_mci[j]
                        elif lable == 0 and lable_cn == 0 and lable_mci == 0:
                            rid_c_value = means_smc[j]

                evidences[i][j] = rid_c_value
    # print(evidences)

    evidences = evidences.astype(np.float32)

    ad_evidences_thr = np.load(evidence_save_path + 'ad_thr.npy')
    cn_evidences_thr = np.load(evidence_save_path + 'cn_thr.npy')

    evidence_nor = []
    evidences_1 = np.zeros(evidences.shape)  # 归一化的数据集
    evidences_ad_abnor = np.zeros(evidences.shape)  # 与ad对比的异常
    evidences_cn_abnor = np.zeros(evidences.shape)  # 与正常人对比的异常

    for i in range(ad.size):
        tmp_sample = evidences[i]
        tmp_nor_distan = [0, 0]

        for j in range(tmp_sample.size):
            c_value = tmp_sample[j]
            evidences_1[i][j] = (c_value - min_array[j]) / gaps[j]
            ad_nor_low = ad_evidences_thr[2 * j]
            ad_nor_high = ad_evidences_thr[2 * j + 1]

            cn_nor_low = cn_evidences_thr[2 * j]
            cn_nor_high = cn_evidences_thr[2 * j + 1]

            if ad_nor_high < ad_nor_low:
                if c_value != ad_nor_low:
                    tmp_nor_distan[0] += 1
                    # evidences_2[i][j]=1
                    evidences_ad_abnor[i][j] = 1
            else:
                if not (c_value >= ad_nor_low and c_value <= ad_nor_high):
                    if c_value < ad_nor_low:
                        if ad_nor_low == 0:
                            tmp_nor_distan[0] += 1
                            # evidences_2
                            evidences_ad_abnor[i][j] = 1
                        else:
                            tmp_nor_distan[0] += (ad_nor_low - c_value + abs(ad_nor_low)) / abs(ad_nor_low)
                            evidences_ad_abnor[i][j] = (ad_nor_low - c_value + abs(ad_nor_low)) / abs(ad_nor_low)
                    if c_value > ad_nor_high:
                        if ad_nor_high == 0:
                            tmp_nor_distan[0] += 1
                            evidences_ad_abnor[i][j] = 1
                        else:
                            tmp_nor_distan[0] += (c_value - ad_nor_high + abs(ad_nor_high)) / abs(ad_nor_high)
                            evidences_ad_abnor[i][j] = (c_value - ad_nor_high + abs(ad_nor_high)) / abs(ad_nor_high)

            if cn_nor_high < cn_nor_low:
                if c_value != cn_nor_low:
                    tmp_nor_distan[1] += 1
                    evidences_cn_abnor[i][j] = 1
            else:
                if not (c_value >= cn_nor_low and c_value <= cn_nor_high):
                    if c_value < cn_nor_low:
                        if cn_nor_low == 0:
                            tmp_nor_distan[1] += 1
                            evidences_cn_abnor[i][j] = 1
                        else:
                            tmp_nor_distan[1] += (cn_nor_low - c_value + abs(cn_nor_low)) / abs(cn_nor_low)
                            evidences_cn_abnor[i][j] = (cn_nor_low - c_value + abs(cn_nor_low)) / abs(cn_nor_low)
                    if c_value > cn_nor_high:
                        if cn_nor_high == 0:
                            tmp_nor_distan[1] += 1
                            evidences_cn_abnor[i][j] = 1
                        else:
                            tmp_nor_distan[1] += (c_value - cn_nor_high + abs(cn_nor_high)) / abs(cn_nor_high)
                            evidences_cn_abnor[i][j] = (c_value - cn_nor_high + abs(cn_nor_high)) / abs(cn_nor_high)

        evidence_nor.append(tmp_nor_distan)

    return np.array(evidence_nor), evidences_1, np.concatenate((evidences_ad_abnor, evidences_cn_abnor), axis=1)

def drawn_distance_cluster(ad, mci, evidences, data_len=14, is_muti_centers=False, center_save_path='',
                           isTraining=True):
    if isTraining:
        if not is_muti_centers:
            ad_num = np.zeros(data_len)
            cn_num = np.zeros(data_len)

            ad_sum = np.zeros(data_len)
            cn_sum = np.zeros(data_len)

            for i in range(ad.size):
                lable = int(ad[i])
                lable_mci = int(mci[i])
                tmp_data = evidences[i]
                for j in range(tmp_data.size):
                    c_value = tmp_data[j]

                    if lable == 1:
                        ad_sum[j] += c_value
                        ad_num[j] += 1
                    elif lable == 0 and lable_mci == 0:
                        cn_sum[j] += c_value
                        cn_num[j] += 1

            means_ad = np.array([ad_sum / ad_num])
            means_cn = np.array([cn_sum / cn_num])

        else:
            ad_sample = []
            cn_sample = []
            for i in range(ad.size):
                if ad[i] == 1:
                    ad_sample.append(evidences[i])
                elif ad[i] == 0 and mci[i] == 0:
                    cn_sample.append(evidences[i])

            mbk_ad = MiniBatchKMeans(init='k-means++', n_clusters=10, max_iter=150, random_state=0, batch_size=512,
                                     n_init=10, max_no_improvement=10, verbose=0)
            mbk_ad.fit(np.array(ad_sample))

            mbk_cn = MiniBatchKMeans(init='k-means++', n_clusters=10, max_iter=150, random_state=0, batch_size=512,
                                     n_init=10, max_no_improvement=10, verbose=0)
            mbk_cn.fit(np.array(cn_sample))

            means_ad = mbk_ad.cluster_centers_
            means_cn = mbk_cn.cluster_centers_

        if os.path.exists(center_save_path + 'ad_centers.npy'):
            os.remove(center_save_path + 'ad_centers.npy')

        if os.path.exists(center_save_path + 'cn_centers.npy'):
            os.remove(center_save_path + 'cn_centers.npy')

        np.save(center_save_path + 'ad_centers.npy', means_ad)
        np.save(center_save_path + 'cn_centers.npy', means_cn)
    else:
        means_ad = np.load(center_save_path + 'ad_centers.npy')
        means_cn = np.load(center_save_path + 'cn_centers.npy')


def get_evidence_from_dic(rids, viscodes, save_path, training=False):
    data_dic = ''
    if len(g_evidence_dic) < 1:
        if os.path.exists(save_path + '_evidences.npy'):
            data_dic = np.load(save_path + '_evidences.npy', allow_pickle=True).item()
            # print('字典已存在!')
            g_evidence_dic.update(data_dic)
        print('加载字典完成，字典长度为：   ', len(g_evidence_dic))
        # print(g_evidence_dic)
    else:
        data_dic = g_evidence_dic

    ret_value = []
    for i in range(rids.size):
        if training:
            rid = rids[i]
            viscode = viscodes[i]
        else:
            rid = str(rids[i], encoding="utf-8")
            viscode = str(viscodes[i], encoding="utf-8")

        key = str(rid) + '_' + str(viscode)

        ret_value.append(data_dic[key])

    return np.array(ret_value)


def get_data_result(y_true, y_pred, output_len):
    ytrue_ret = []
    ypred_ret = []

    # np.zeros(result_tr[12].size)

    n_output = len(y_true)

    for i in range(output_len):
        ytrue_ret.append([])

    data_len = len(y_pred)

    for i in range(data_len):
        tmp_pred = np.zeros(output_len)
        for j in range(n_output):
            if j < output_len:
                ytrue_ret[j].append(y_true[j][i])
                tmp_pred[j] = y_pred[i][j]
            else:
                ytrue_ret[output_len - 1][i] += y_true[j][i]
                tmp_pred[output_len - 1] += y_pred[i][j]
        ypred_ret.append(tmp_pred)
    return ytrue_ret, ypred_ret


def get_one_hot_index(data, index):
    ret_value = []
    for i in range(len(data)):
        ret_value.append(data[i][index])
    return ret_value


def get_score_final(ys_true, openmax_rate=[], thr=0.95):
    nb_outputs = len(ys_true)
    ys_pred_open_max = []

    if len(openmax_rate) > 0:
        for i in range(nb_outputs):
            ys_pred_open_max.append([])

    for j in range(len(openmax_rate)):
        tmp_open_max = openmax_rate[j]
        index = np.argmax(tmp_open_max)

        thr_low = False
        for l in range(nb_outputs):
            if index == (nb_outputs - 1):
                if l == index:
                    ys_pred_open_max[l].append(1.0)
                else:
                    ys_pred_open_max[l].append(0.0)
            else:
                if l == index:
                    if tmp_open_max[l] >= thr:
                        ys_pred_open_max[l].append(1.0)
                        thr_low = True
                    else:
                        ys_pred_open_max[l].append(0.0)
                else:
                    if l == (nb_outputs - 1):
                        if thr_low:
                            ys_pred_open_max[l].append(0.0)
                        else:
                            ys_pred_open_max[l].append(1.0)
                    else:
                        ys_pred_open_max[l].append(0.0)

    retValue = []

    for i in range(0, nb_outputs):
        c_true = ys_true[i]
        c_pred = ys_pred_open_max[i]

        sensitivity_c = 0
        fpr = 0

        f_unkown = 0

        for k in range(len(c_pred)):
            true_value = c_true[k]
            pred_value = c_pred[k]

            if true_value == 1 and pred_value == 1:
                sensitivity_c += 1

            if (i == 0 or i == 1) and true_value == 1 and pred_value == 0 and ys_pred_open_max[2][k]:
                f_unkown += 1

            if true_value == 0 and pred_value == 1:
                fpr += 1

        retValue.append([sensitivity_c / sum(c_true), fpr / (len(c_true) - sum(c_true)), f_unkown / sum(c_true)])

    acc = 0
    for i in range(0, len(ys_true[0])):
        iscorrect = True
        for j in range(nb_outputs):
            if ys_true[j][i] != ys_pred_open_max[j][i]:
                iscorrect = False
        if iscorrect:
            acc += 1

    return acc / len(ys_true[0]), retValue


def get_known2unknown_2v(m_rid, m_viscode, methond, ys_true, openmax_rate=[], thr=0.95):
    nb_outputs = len(ys_true)
    ys_pred_open_max = []

    if len(openmax_rate) > 0:
        for i in range(nb_outputs):
            ys_pred_open_max.append([])

    for j in range(len(openmax_rate)):
        tmp_open_max = openmax_rate[j]
        index = np.argmax(tmp_open_max)

        thr_low = False
        for l in range(nb_outputs):
            if index == (nb_outputs - 1):
                if l == index:
                    ys_pred_open_max[l].append(1.0)
                else:
                    ys_pred_open_max[l].append(0.0)
            else:
                if l == index:
                    if tmp_open_max[l] >= thr:
                        ys_pred_open_max[l].append(1.0)
                        thr_low = True
                    else:
                        ys_pred_open_max[l].append(0.0)
                else:
                    if l == (nb_outputs - 1):
                        if thr_low:
                            ys_pred_open_max[l].append(0.0)
                        else:
                            ys_pred_open_max[l].append(1.0)
                    else:
                        ys_pred_open_max[l].append(0.0)

    ret_str = ''

    for i in range(0, nb_outputs):
        c_true = ys_true[i]
        c_pred = ys_pred_open_max[i]

        for k in range(len(c_pred)):
            true_value = c_true[k]
            pred_value = c_pred[k]
            if (i == 0 or i == 1) and true_value == 1 and pred_value == 1:
                tmp_str = ''
                tmp_str += m_rid[k] + ','
                tmp_str += m_viscode[k] + ','
                tmp_methond = methond[k]
                for j in range(tmp_methond.size):
                    tmp_str += str(int(tmp_methond[j])) + ','
                tmp_str += str(int(ys_true[0][k])) + ','
                tmp_str += str(int(ys_true[1][k])) + ','
                tmp_str += str(format(openmax_rate[k][0], '.4f')) + ','
                tmp_str += str(format(openmax_rate[k][1], '.4f')) + '\n'
                ret_str += tmp_str

    with open('/data/huangyunyou/result/correct_subject.txt', 'a') as f:
        f.write(ret_str)


def get_methond_count(methond, y_label, nb_output):
    methond_set = []

    methond_total = set()

    for i in range(nb_output):
        methond_set.append(set())

    for i in range(len(methond)):
        tmp_methond = methond[i]

        m_key = ''
        for j in range(tmp_methond.size):
            if tmp_methond[j] == 1:
                m_key += '1_'
            else:
                m_key += '0_'

        methond_total.add(m_key)

        for k in range(nb_output):
            if y_label[k][i] == 1:
                methond_set[k].add(m_key)
    m_len = []
    for j in range(nb_output):
        m_len.append(len(methond_set[j]))

    return len(methond_total), m_len

#

train_strategy = 1

# Train the model to classify AD and CN
if train_strategy==1:
    # 指定日志目录
    log_dir = "/data/huangyunyou/ODMLCS_MODEL/logs/" + datetime.datetime.now().strftime("%Y%m%d") + '/' + str(
        acm_learning_rate) + '_' + str(acm_batch_size) + '_' + model_version + '/'
    summary_writer = tf.summary.create_file_writer(log_dir)  # 创建日志文件句柄

    train_loss_file_path = log_dir + 'ae_ac_train_loss.txt'
    eval_loss_file_path = log_dir + 'ae_ac_eval_loss.txt'
    eval_auc_file_path = log_dir + 'ae_ac_eval_auc.txt'

    AE_model = HierarchicalOpenNet.HierarchicalOpenNet(is_continue_train=True,save_path='/data/huangyunyou/ODMLCS_MODEL/save_models/ae_ac_0.0005_32_210v')

    ae_train_count = 0

    min_means_loss_ae = 100000
    min_var_loss_ae = 100000
    min_loss_ae = 100000

    ae_histy_tr = []
    ae_histy_va = []

    ae_early_flage = False
    while (True):
        ae_train_count += 1
        data, ad, cn, rid, viscode = ac_tr_it.get_next()
        batch, x, y = data.shape
        data = tf.reshape(data, [batch, -1, 2090])
        tf.keras.backend.set_learning_phase(True)
        # data=pad_sequences(data, maxlen=steps, dtype='float', value=-4.0)
        hist = AE_model.train_on_batch_customer(data, [get_one_hot([ad.numpy(),cn.numpy()],batch),data], return_dict=True)
        with open(train_loss_file_path, 'a') as f:
            f.write(str(hist['loss']) + ',' + str(ae_train_count) + '\n')
        with summary_writer.as_default():  # 将loss写入TensorBoard
            tf.summary.scalar('ae_train_loss', hist['loss'], step=ae_train_count)
            summary_writer.flush()
        ae_histy_tr.append(hist)

        if ae_train_count % 100 == 0:
            data_va, ad_va, cn_va, rid_va, viscode_va = ac_va_it.get_next()
            batch, x, y = data_va.shape
            # data_va = tf.reshape(data_va, [x, -1, 2090])
            data_va = tf.reshape(data_va, [batch, -1, 2090])
            # data_va = pad_sequences(data_va, maxlen=steps, dtype='float', value=-4.0)
            tf.keras.backend.set_learning_phase(False)

            hist_va = AE_model.test_on_batch_customer(data_va, [get_one_hot([ad_va.numpy(),cn_va.numpy()],batch),data_va], return_dict=True)
            with open(eval_loss_file_path, 'a') as f:
                f.write(str(hist_va['loss']) + ',' + str(ae_train_count) + '\n')
            with summary_writer.as_default():  # 将loss写入TensorBoard
                tf.summary.scalar('ae_eval_loss', hist_va['loss'], step=ae_train_count)
                summary_writer.flush()
            print('AE model validation loss ============   ' + str(hist_va['loss']) + '       steps ======== ' + str(
                ae_train_count))

            ae_histy_va.append(hist_va)

            means, var, current_value = if_model_save(ae_histy_va, 'loss', 3)
            if means < min_means_loss_ae and current_value * 0.95 < min_loss_ae:
                # min_loss_acm = hist_va['loss']
                min_means_loss_ae = means
                min_var_loss_ae = var
                min_loss_ae = current_value
                print('########################   Save model    ', ae_train_count)
                AE_model.save_model(model_save_path + 'save_models/ae_ac_' + str(acm_learning_rate) + '_' + str(
                    acm_batch_size) + '_' + model_version)
            ys_pred = AE_model.predict_customer(data_va)
            # print('********************')
            # print(ys_true)
            acm_auc = get_auc([ad_va, cn_va], get_divided(ys_pred[5], batch, 2))

            auc_str = ''
            for i in range(len(acm_auc)):
                tmp_auc = acm_auc[i]
                auc_str += str(tmp_auc) + ','
                with summary_writer.as_default():  # 将loss写入TensorBoard
                    tf.summary.scalar('ac_auc_' + str(i), tmp_auc, step=ae_train_count)
                    summary_writer.flush()
            auc_str += str(ae_train_count)
            with open(eval_auc_file_path, 'a') as f:
                f.write(auc_str + '\n')
            print('Ac model validation AUC ============   ' + str(acm_auc) + '       steps ======== ' + str(
                ae_train_count))

            ae_early_flag = early_stop(ae_histy_va, 'loss', 0.0001, 180)
            if ae_early_flag:
                break

# Obtain the abnormal patterns of subject
elif train_strategy == 2:
    ac_tr = []
    ac_va = []
    ac_te = []

    ac_tr_pre = get_data_test_set(ac_training_set, sp_read_and_decode, acm_batch_size, 1)
    ac_tr_it_pre = ac_tr_pre.as_numpy_iterator()
    ac_va_pre = get_data_test_set(ac_validation_set, sp_read_and_decode, acm_batch_size, 1)
    ac_va_it_pre = ac_va_pre.as_numpy_iterator()
    ac_te_pre = get_data_test_set(ac_test_set, sp_read_and_decode, acm_batch_size, 1)
    ac_te_it_pre = ac_te_pre.as_numpy_iterator()

    mci_te_pre = get_data_test_set(mci_test_set, sp_read_and_decode, acm_batch_size, 1)
    mci_te_it_pre = mci_te_pre.as_numpy_iterator()

    smc_te_pre = get_data_test_set(smc_test_set, acm_read_and_decode, acm_batch_size)
    smc_te_it_pre = smc_te_pre.as_numpy_iterator()

    count = 0
    for data, ad, cn, rid, viscode in ac_tr_it_pre:
        count += 1
        batch, x, y = data.shape
        core_evidence = get_core_evidence_4v(rid, viscode, batch, training=False)
        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid]
        tmp_reslut.append(viscode)  # 15  unkown lable
        tmp_reslut.append(core_evidence)  # evidence             8
        tmp_reslut.append(ad)  # 13
        tmp_reslut.append(cn)  # 13
        tmp_reslut.append(np.zeros(batch))  # 13
        # tmp_reslut.append(np.zeros(batch))  #                  15  unkown lable

        ac_tr.append(tmp_reslut)
        if count % 50 == 0:
            print('index   ', count)
        # if count>500:
        #    break
    result_tr = concate_result_5v(ac_tr)

    print('#########################')
    c_evidences, c_rids, c_ads, c_cns, c_mcis, c_viscodes = clear_evidence_2v(result_tr[0], result_tr[1], result_tr[2],
                                                                              result_tr[3], result_tr[4], result_tr[5])

    # core_evidence_1=get_core_evidence_convert(c_rids,c_ads,copy.copy(c_evidences),evidence_save_path='/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/draw_tr_core_evidence_14_core',data_len=14)
    core_abnor, _, core_abnor_value = get_core_evidence_convert_abnormal(c_rids, c_ads, copy.copy(c_evidences),
                                                                         evidence_save_path='/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_tr_core_evidence_14_core_abnor_adcn_mcismc_',
                                                                         data_len=14)

    # drawn_distance_cluster(c_ads, np.zeros(c_ads.shape), core_evidence_1, data_len=14, is_muti_centers=True,center_save_path='/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/muti_')

    ac_tr = []
    count = 0
    for data, ad, cn, rid, viscode in ac_va_it_pre:
        count += 1
        batch, x, y = data.shape
        core_evidence = get_core_evidence_4v(rid, viscode, batch, training=False)
        tmp_reslut = [rid]
        tmp_reslut.append(viscode)  # 15  unkown lable
        tmp_reslut.append(core_evidence)  # evidence             8
        tmp_reslut.append(ad)  # 13
        tmp_reslut.append(cn)  # 13
        tmp_reslut.append(np.zeros(batch))  # 15  unkown lable

        ac_tr.append(tmp_reslut)

        # if count >= 1000:
        #    count = 0
        #    break

    for data, ad, cn, rid, viscode in ac_te_it_pre:
        count += 1
        batch, x, y = data.shape
        core_evidence = get_core_evidence_4v(rid, viscode, batch, training=False)
        tmp_reslut = [rid]
        tmp_reslut.append(viscode)  # 15  unkown lable
        tmp_reslut.append(core_evidence)  # evidence             8
        tmp_reslut.append(ad)  # 13
        tmp_reslut.append(cn)  # 13
        tmp_reslut.append(np.zeros(batch))  # 15  unkown lable

        ac_tr.append(tmp_reslut)

        # if count >= 1000:
        #    count = 0
        #    break

    for data, ad, cn, rid, viscode in mci_te_it_pre:
        count += 1
        batch, x, y = data.shape
        core_evidence = get_core_evidence_4v(rid, viscode, batch, training=False)

        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid]
        tmp_reslut.append(viscode)  # 15  unkown lable
        tmp_reslut.append(core_evidence)  # evidence             8
        # print('#####################################################')
        # print(tmp_reslut)
        tmp_reslut.append(ad)  # 13
        tmp_reslut.append(cn)  # 13
        tmp_reslut.append(np.ones(batch))  # 15  unkown lable

        ac_tr.append(tmp_reslut)

        # if count >= 1000:
        #    break

    for data, ad, cn, mci, rid, viscode in smc_te_it_pre:
        count += 1
        batch, x, y = data.shape
        core_evidence = get_core_evidence_4v(rid, viscode, batch, training=False)

        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid]
        tmp_reslut.append(viscode)  # 15  unkown lable
        tmp_reslut.append(core_evidence)  # evidence             8
        # print('#####################################################')
        # print(tmp_reslut)
        tmp_reslut.append(ad)  # 13
        tmp_reslut.append(cn)  # 13
        tmp_reslut.append(mci)  # 15  unkown lable

        ac_tr.append(tmp_reslut)

    result_tr_2v = concate_result_5v(ac_tr)
    c_evidences_2v, c_rids_2v, c_ads_2v, c_cns_2v, c_mcis_2v, c_viscodes_2v = clear_evidence_2v(
        result_tr_2v[0], result_tr_2v[1], result_tr_2v[2],
        result_tr_2v[3], result_tr_2v[4], result_tr_2v[5])
    # core_evidence_2 = get_core_evidence_convert_2v(result_tr_2v[0], result_tr_2v[3], result_tr_2v[4], copy.copy(result_tr_2v[2]),evidence_save_path='/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/draw_tr_core_evidence_14_core',data_len=14)
    core_abnor_2, _, core_abnor_value_2 = get_core_evidence_convert_abnormal_2v_adcn_mcismc(c_rids_2v, c_ads_2v,
                                                                                            c_cns_2v, c_mcis_2v,
                                                                                            copy.copy(c_evidences_2v),
                                                                                            evidence_save_path='/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_tr_core_evidence_14_core_abnor_adcn_mcismc_',
                                                                                            data_len=14, )
    drawn_distance_cluster(c_ads, np.zeros(c_ads.shape), core_abnor_value, 28, True,
                           '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_abnor_value_Muti_adcn_mcismc_')

# Train the model to classify AD and CN in open clinical setting
elif train_strategy == 3:
    ac_tr = []
    ac_va = []
    ac_te = []

    ac_tr_pre = get_data_test_set(ac_training_set, sp_read_and_decode, acm_batch_size, 1)
    ac_tr_it_pre = ac_tr_pre.as_numpy_iterator()
    ac_va_pre = get_data_test_set(ac_validation_set, sp_read_and_decode, acm_batch_size, 1)
    ac_va_it_pre = ac_va_pre.as_numpy_iterator()
    ac_te_pre = get_data_test_set(ac_test_set, sp_read_and_decode, acm_batch_size, 1)
    ac_te_it_pre = ac_te_pre.as_numpy_iterator()

    mci_te_pre = get_data_test_set(mci_test_set, sp_read_and_decode, acm_batch_size, 1)
    mci_te_it_pre = mci_te_pre.as_numpy_iterator()

    smc_te_pre = get_data_test_set(smc_test_set, acm_read_and_decode, acm_batch_size)
    smc_te_it_pre = smc_te_pre.as_numpy_iterator()

    # AE_model_pre = keras.models.load_model('/data/huangyunyou/ODMLCS_MODEL/save_models/ae_0.0005_32_22v.h5',custom_objects=_custom_objects)

    # AE_model = HierarchicalOpenNet_12v.HierarchicalOpenNet_12v(is_continue_train=True,save_path='/data/huangyunyou/ODMLCS_MODEL/save_models/ac_0.0005_64_820v')
    AE_model = HierarchicalOpenNet.HierarchicalOpenNet(is_continue_train=True,
                                                       save_path='/data/huangyunyou/ODMLCS_MODEL/save_models/ae_ac_0.0005_32_210v')
    tf.keras.backend.set_learning_phase(False)

    # count=0

    for data, ad, cn, rid, viscode in ac_tr_it_pre:
        data = tf.reshape(data, [-1, steps, 2090])
        # data = pad_sequences(data, maxlen=steps, dtype='float', value=-4.0)
        batch, x, y = data.shape
        data = tf.reshape(data, [batch, -1, 2090])
        core_evidence = get_evidence_from_dic(rid, viscode,
                                              '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_abnor_adcn_mcismc_')[
                        :, 0:28]  # 证据

        predict_result = AE_model.predict_customer(data)  # z,layer1,layer2,layer3,layer4,pred,loss
        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid, viscode, methond_simplified(data.numpy()[:, :, 0:60])]
        tmp_reslut.append(predict_result[4])  # 4                    3
        tmp_reslut.append(core_evidence)  # evidence                 4
        tmp_reslut.append(predict_result[5])  # pred(one-hot)        5
        tmp_reslut.append(predict_result[6])  # loss                 6
        tmp_reslut.append(get_singel_label([ad, cn], batch, 2))  # 7
        # print('#####################################################')
        # print(tmp_reslut)
        tmp_reslut.append(ad)  # 8
        tmp_reslut.append(cn)  # 9
        tmp_reslut.append(np.zeros(batch))  # 10  unkown lable
        ac_tr.append(tmp_reslut)

    result_tr = concate_result_3v(ac_tr, muti_valu_index=5)

    print('Start train weibull model ..... ..... ..... ..... ..... ..... ..... ')

    # '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/mnist_av.h5'
    # '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/mnist_fashion_weibull_model.pkl'

    # weibull_fit('/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/ac_0.0005_64_820v_14_core_evidence_weibull_model.pkl',result_tr[8],result_tr[10],result_tr[12])
    core_evidence_weibull_model = weibull_fit_2v(
        '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/ac_0.0005_32_210v_28_core_evidence_abnor_weibull_model_adcn_mcismc_800tail.pkl',
        result_tr[4], result_tr[5], result_tr[7],
        '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_abnor_value_Muti_adcn_mcismc_')

    print('Complete train core_evidence model ..... ..... ..... ..... ..... ..... ..... ')

    means_abnor_value_ad = np.load(
        '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_abnor_value_Muti_adcn_mcismc_' + 'ad_centers.npy')
    means_abnor_value_cn = np.load(
        '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_abnor_value_Muti_adcn_mcismc_' + 'cn_centers.npy')

    weibull_ad_thr = np.percentile(query_weibull(0, core_evidence_weibull_model)[1][0], 90)
    weibull_cn_thr = np.percentile(query_weibull(1, core_evidence_weibull_model)[1][0], 90)
    print('ad_thr  ', weibull_ad_thr)
    print('cn_thr  ', weibull_cn_thr)

    score = result_tr[3]

    '''
    openmax_result = get_divided(result_tr[5], result_tr[5].shape[0], 2)

    get_methonds_2v([result_tr[8], result_tr[9]], openmax_result,result_tr[0], result_tr[1], result_tr[2],
                    model_save_path + 'save_models/ac_0.0005_32_210v_closed_set_train_adcn_mcismc')
    print('train_stage   ')
    print('training closed_set ',statistics_transform)
    '''

    result_y4 = []
    for i in range(score.shape[0]):
        openmax, softmax = recalibrate_scores_4v(score[i, :], core_evidence_weibull_model,
                                                 core_evidence_weibull_model.keys(), alpha_rank=2,
                                                 center_path='Muti', isLastLayer=False, real_acv=result_tr[4][i, :],
                                                 means_ad=means_abnor_value_ad, means_cn=means_abnor_value_cn,
                                                 score_cal_type=1, modify=1,
                                                 weibull_ad_thr=weibull_ad_thr, weibull_cn_thr=weibull_cn_thr)

        result_y4.append(openmax)
    openmax_result = get_divided(result_y4, len(result_y4), 3)

    get_methonds_2v([result_tr[8], result_tr[9], result_tr[10]], openmax_result, result_tr[0], result_tr[1],
                    result_tr[2],
                    model_save_path + 'save_models/ac_0.0005_32_210v_open_set_evidence_train_adcn_mcismc_2v')
    print('training evidence ', statistics_transform)

    ac_tr = []
    for data, ad, cn, rid, viscode in ac_va_it_pre:
        data = tf.reshape(data, [-1, steps, 2090])
        # data = pad_sequences(data, maxlen=steps, dtype='float', value=-4.0)
        batch, x, y = data.shape
        data = tf.reshape(data, [batch, -1, 2090])
        core_evidence = get_evidence_from_dic(rid, viscode,
                                              '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_abnor_adcn_mcismc_')[
                        :, 0:28]  # 证据

        predict_result = AE_model.predict_customer(data)  # z,layer1,layer2,layer3,layer4,pred,loss
        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid, viscode, methond_simplified(data.numpy()[:, :, 0:60])]
        tmp_reslut.append(predict_result[4])  # 4                    3
        tmp_reslut.append(core_evidence)  # evidence                 4
        tmp_reslut.append(predict_result[5])  # pred(one-hot)        5
        tmp_reslut.append(predict_result[6])  # loss                 6
        tmp_reslut.append(get_singel_label([ad, cn], batch, 2))  # 7
        # print('#####################################################')
        # print(tmp_reslut)
        tmp_reslut.append(ad)  # 8
        tmp_reslut.append(cn)  # 9
        tmp_reslut.append(np.zeros(batch))  # 10  unkown lable
        ac_va.append(tmp_reslut)

    result_va = concate_result_3v(ac_va, muti_valu_index=5)

    score_va = result_va[3]

    '''
    openmax_result_va = get_divided(result_va[5], result_va[5].shape[0], 2)

    get_methonds_2v([result_va[8], result_va[9]], openmax_result_va, result_va[0], result_va[1],
                    result_va[2],
                    model_save_path + 'save_models/ac_0.0005_32_210v_closed_set_validation_adcn_mcismc')
    print('validation_stage   ')
    print('closed_set  ', statistics_transform)
    '''

    result_y4_va = []

    for i in range(score_va.shape[0]):
        openmax_va, softmax_va = recalibrate_scores_4v(score_va[i, :], core_evidence_weibull_model,
                                                       core_evidence_weibull_model.keys(), alpha_rank=2,
                                                       isLastLayer=False, real_acv=result_va[4][i, :],
                                                       center_path='muti',
                                                       means_ad=means_abnor_value_ad, means_cn=means_abnor_value_cn,
                                                       score_cal_type=1, modify=1,
                                                       weibull_ad_thr=weibull_ad_thr, weibull_cn_thr=weibull_cn_thr)
        result_y4_va.append(openmax_va)
    openmax_result_va = get_divided(result_y4_va, len(result_y4_va), 3)

    get_methonds_2v([result_va[8], result_va[9], result_va[10]], openmax_result_va, result_va[0], result_va[1],
                    result_va[2],
                    model_save_path + 'save_models/ac_0.0005_32_210v_open_set_evidence_validation_adcn_mcismc_2v')
    # print('train_stage   ')
    print('evidence ', statistics_transform)

    ac_va = []
    for data, ad, cn, rid, viscode in ac_te_it_pre:
        data = tf.reshape(data, [-1, steps, 2090])
        # data = pad_sequences(data, maxlen=steps, dtype='float', value=-4.0)
        batch, x, y = data.shape
        data = tf.reshape(data, [batch, -1, 2090])
        core_evidence = get_evidence_from_dic(rid, viscode,
                                              '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_abnor_adcn_mcismc_')[
                        :, 0:28]  # 证据

        predict_result = AE_model.predict_customer(data)  # z,layer1,layer2,layer3,layer4,pred,loss
        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid, viscode, methond_simplified(data.numpy()[:, :, 0:60])]
        tmp_reslut.append(predict_result[4])  # 4                    3
        tmp_reslut.append(core_evidence)  # evidence                 4
        tmp_reslut.append(predict_result[5])  # pred(one-hot)        5
        tmp_reslut.append(predict_result[6])  # loss                 6
        tmp_reslut.append(get_singel_label([ad, cn], batch, 2))  # 7
        # print('#####################################################')
        # print(tmp_reslut)
        tmp_reslut.append(ad)  # 8
        tmp_reslut.append(cn)  # 9
        tmp_reslut.append(np.zeros(batch))  # 10  unkown lable
        ac_te.append(tmp_reslut)

    for data, ad, cn, rid, viscode in mci_te_it_pre:
        data = tf.reshape(data, [-1, steps, 2090])
        # data = pad_sequences(data, maxlen=steps, dtype='float', value=-4.0)
        batch, x, y = data.shape
        data = tf.reshape(data, [batch, -1, 2090])
        core_evidence = get_evidence_from_dic(rid, viscode,
                                              '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_abnor_adcn_mcismc_')[
                        :, 0:28]  # 证据

        predict_result = AE_model.predict_customer(data)  # z,layer1,layer2,layer3,layer4,pred,loss
        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid, viscode, methond_simplified(data.numpy()[:, :, 0:60])]
        tmp_reslut.append(predict_result[4])  # 4                    3
        tmp_reslut.append(core_evidence)  # evidence                 4
        tmp_reslut.append(predict_result[5])  # pred(one-hot)        5
        tmp_reslut.append(predict_result[6])  # loss                 6
        tmp_reslut.append(get_singel_label([ad, cn], batch, 2))  # 7
        # print('#####################################################')
        # print(tmp_reslut)
        tmp_reslut.append(ad)  # 8
        tmp_reslut.append(cn)  # 9
        tmp_reslut.append(np.ones(batch))  # 10  unkown lable
        ac_te.append(tmp_reslut)

    for data, ad, cn, mci, rid, viscode in smc_te_it_pre:
        data = tf.reshape(data, [-1, steps, 2090])
        # data = pad_sequences(data, maxlen=steps, dtype='float', value=-4.0)
        batch, x, y = data.shape
        data = tf.reshape(data, [batch, -1, 2090])
        core_evidence = get_evidence_from_dic(rid, viscode,
                                              '/data/huangyunyou/ODMLCS_MODEL/save_mid_open_set/5-95_abnor_adcn_mcismc_')[
                        :, 0:28]  # 证据

        predict_result = AE_model.predict_customer(data)  # z,layer1,layer2,layer3,layer4,pred,loss
        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid, viscode, methond_simplified(data.numpy()[:, :, 0:60])]
        tmp_reslut.append(predict_result[4])  # 4                    3
        tmp_reslut.append(core_evidence)  # evidence                 4
        tmp_reslut.append(predict_result[5])  # pred(one-hot)        5
        tmp_reslut.append(predict_result[6])  # loss                 6
        tmp_reslut.append(get_singel_label([ad, cn], batch, 2))  # 7
        # print('#####################################################')
        # print(tmp_reslut)
        tmp_reslut.append(ad)  # 8
        tmp_reslut.append(cn)  # 9
        tmp_reslut.append(mci)  # 10  unkown lable
        ac_te.append(tmp_reslut)

    result_te = concate_result_3v(ac_te, muti_valu_index=5)

    score_te = result_te[3]

    '''
    openmax_result_te = get_divided(result_te[5], result_te[5].shape[0], 2)

    get_methonds_2v([result_te[8], result_te[9]], openmax_result_te, result_te[0], result_te[1],
                    result_te[2],
                    model_save_path + 'save_models/ac_0.0005_32_210v_closed_set_test_adcn_mcismc',is_test=True)
    print('test_stage   ')
    print('closed_set   ',statistics_transform)
    '''
    result_y4_te = []

    for i in range(score_te.shape[0]):
        openmax_te, softmax_te = recalibrate_scores_4v(score_te[i, :], core_evidence_weibull_model,
                                                       core_evidence_weibull_model.keys(), alpha_rank=2,
                                                       isLastLayer=False, real_acv=result_te[4][i, :],
                                                       center_path='muti',
                                                       means_ad=means_abnor_value_ad, means_cn=means_abnor_value_cn,
                                                       score_cal_type=1, modify=1,
                                                       weibull_ad_thr=weibull_ad_thr, weibull_cn_thr=weibull_cn_thr
                                                       )
        result_y4_te.append(openmax_te)
    openmax_result_te = get_divided(result_y4_te, len(result_y4_te), 3)

    get_methonds_2v([result_te[8], result_te[9], result_te[10]], openmax_result_te, result_te[0], result_te[1],
                    result_te[2],
                    model_save_path + 'save_models/ac_0.0005_32_210v_open_set_evidence_test_adcn_mcismc_2v',
                    is_test=True)
    # print('train_stage   ')
    print('evidence ', statistics_transform)

# Train the model to select the exams for subject
elif train_strategy == 4:
    methond_prediction_model = get_methond_model_4v()
    methond_train_model = get_trainable_model_methond(methond_prediction_model, loss_type='M_BCE',
                                                      methond_list=[43967, 38837, 24215, 26966, 34630, 14586, 3425,
                                                                    8966, 3255, 1524, 1910, 610])
    methond_train_model.summary()
    # plot_model(acm_train_model, to_file=model_save_path+'save_models_image/acm_model.png',show_shapes=True)
    methond_train_model.compile(optimizer=methond_opt_adam)

    # 指定日志目录
    log_dir = "/data/huangyunyou/ODMLCS_MODEL/logs/" + datetime.datetime.now().strftime("%Y%m%d") + '/' + str(
        methond_learning_rate) + '_' + str(methond_batch_size) + '_' + model_version + '/'
    summary_writer = tf.summary.create_file_writer(log_dir)  # 创建日志文件句柄

    train_loss_file_path = log_dir + 'ac_methond_train_loss.txt'
    eval_loss_file_path = log_dir + 'ac_methond_eval_loss.txt'
    eval_auc_file_path = log_dir + 'ac_methond_eval_auc.txt'

    # model_version='1v'

    methond_train_count = 0

    min_loss_methond = 100000
    min_var_loss_methond = 100000
    min_means_loss_methond = 100000

    methond_histy_tr = []
    methond_histy_va = []

    methond_early_flage = False

    # tr_methond_trans,ev_methond_trans=get_methond_trans_dic()

    while (True):
        methond_train_count += 1
        data, ad, cn, rid, viscode = ac_tr_it.get_next()
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print(ad)
        batch, x, y = data.shape
        data = tf.reshape(data, [batch, -1, 2090])
        methond_data = methond_simplified(data.numpy()[:, :, 0:60])

        methond_and_pred = get_methond_batch(rid, viscode, methond_data,
                                             model_save_path + 'save_models/ac_0.0005_32_210v_open_set_evidence_train_adcn_mcismc_2v',
                                             dic_type=0)
        # methond_and_pred = get_methond_batch_2v(rid, viscode, methond_data,tr_methond_trans)

        tf.keras.backend.set_learning_phase(True)
        # data=pad_sequences(data, maxlen=steps, dtype='float', value=-4.0)
        # print(data.shape)
        # print(methond_and_pred[12].shape)
        # print(methond_and_pred[13].shape)
        # print(methond_and_pred[14].shape)

        # smax=np.concatenate((methond_and_pred[12].reshape(batch,1),methond_and_pred[13].reshape(batch,1),methond_and_pred[14].reshape(batch,1)),axis=1)

        # for i in range(batch):
        #    print('####     ',methond_and_pred[12][i],'  ',methond_and_pred[13][i],'  ',methond_and_pred[14][i],'         ',compute_softmax_probability(smax[i]))

        hist = methond_train_model.train_on_batch(
            [data, methond_and_pred[12], methond_and_pred[13], methond_and_pred[14],
             methond_and_pred[0],
             methond_and_pred[1],
             methond_and_pred[2],
             methond_and_pred[3],
             methond_and_pred[4],
             methond_and_pred[5],
             methond_and_pred[6],
             methond_and_pred[7],
             methond_and_pred[8],
             methond_and_pred[9],
             methond_and_pred[10],
             methond_and_pred[11]
             ],
            return_dict=True)
        with open(train_loss_file_path, 'a') as f:
            f.write(str(hist['loss']) + ',' + str(methond_train_count) + '\n')
        with summary_writer.as_default():  # 将loss写入TensorBoard
            tf.summary.scalar('ac_methond_train_loss', hist['loss'], step=methond_train_count)
            summary_writer.flush()
        methond_histy_tr.append(hist)

        if methond_train_count % 100 == 0:
            data_va, ad_va, cn_va, rid_va, viscode_va = ac_va_it.get_next()
            batch, x, y = data_va.shape
            # data_va = tf.reshape(data_va, [x, -1, 2090])
            data_va = tf.reshape(data_va, [batch, -1, 2090])
            methond_data_va = methond_simplified(data_va.numpy()[:, :, 0:60])
            # data_va = pad_sequences(data_va, maxlen=steps, dtype='float', value=-4.0)
            methond_and_pred_va = get_methond_batch(rid_va, viscode_va, methond_data_va,
                                                    model_save_path + 'save_models/ac_0.0005_32_210v_open_set_evidence_validation_adcn_mcismc_2v',
                                                    dic_type=1)
            # methond_and_pred_va = get_methond_batch_2v(rid_va, viscode_va, methond_data_va,ev_methond_trans)

            tf.keras.backend.set_learning_phase(False)

            hist_va = methond_train_model.test_on_batch(
                [data_va, methond_and_pred_va[12], methond_and_pred_va[13], methond_and_pred_va[14],
                 methond_and_pred_va[0],
                 methond_and_pred_va[1],
                 methond_and_pred_va[2],
                 methond_and_pred_va[3],
                 methond_and_pred_va[4],
                 methond_and_pred_va[5],
                 methond_and_pred_va[6],
                 methond_and_pred_va[7],
                 methond_and_pred_va[8],
                 methond_and_pred_va[9],
                 methond_and_pred_va[10],
                 methond_and_pred_va[11]],
                return_dict=True)
            with open(eval_loss_file_path, 'a') as f:
                f.write(str(hist_va['loss']) + ',' + str(methond_train_count) + '\n')
            with summary_writer.as_default():  # 将loss写入TensorBoard
                tf.summary.scalar('ac_methond_eval_loss', hist_va['loss'], step=methond_train_count)
                summary_writer.flush()
            print(
                'Ac_Methond model validation loss ============   ' + str(
                    hist_va['loss']) + '       steps ======== ' + str(
                    methond_train_count))

            methond_histy_va.append(hist_va)

            means, var, current_value = if_model_save(methond_histy_va, 'loss', 3)
            if means < min_means_loss_methond and current_value * 0.95 < min_loss_methond:
                # min_loss_methond = hist_va['loss']
                min_means_loss_methond = means
                min_var_loss_methond = var
                min_loss_methond = current_value
                print('########################   Save model    ', methond_train_count)
                methond_train_model.save(
                    model_save_path + 'save_models/ac_210v_open_set_evidence_methond_adcn_mcismc_' + str(
                        methond_learning_rate) + '_' + str(
                        methond_batch_size) + '_' + model_version + '.h5')
            methond_histy_va.append(hist_va)

            ys_true, ys_pred = methond_train_model.predict_on_batch(
                [data_va, methond_and_pred_va[12], methond_and_pred_va[13], methond_and_pred_va[14],
                 methond_and_pred_va[0],
                 methond_and_pred_va[1],
                 methond_and_pred_va[2],
                 methond_and_pred_va[3],
                 methond_and_pred_va[4],
                 methond_and_pred_va[5],
                 methond_and_pred_va[6],
                 methond_and_pred_va[7],
                 methond_and_pred_va[8],
                 methond_and_pred_va[9],
                 methond_and_pred_va[10],
                 methond_and_pred_va[11]])
            # print('********************')
            # print(ys_true)

            # for i in range(batch):
            #    print('ture    ',ys_true[0][i],',',ys_true[1][i],',',ys_true[2][i],',',ys_true[3][i],',',ys_true[4][i],',',ys_true[5][i],',',ys_true[6][i],',',ys_true[7][i],',',ys_true[8][i],',',ys_true[9][i],',',ys_true[10][i],',',ys_true[11][i])
            #    print('pred    ',ys_pred[0][i],',',ys_pred[1][i],',',ys_pred[2][i],',',ys_pred[3][i],',',ys_pred[4][i],',',ys_pred[5][i],',',ys_pred[6][i],',',ys_pred[7][i],',',ys_pred[8][i],',',ys_pred[9][i],',',ys_pred[10][i],',',ys_pred[11][i])
            methond_auc = get_auc(ys_true, ys_pred)

            auc_str = ''
            for i in range(len(methond_auc)):
                tmp_auc = methond_auc[i]
                auc_str += str(tmp_auc) + ','
                with summary_writer.as_default():  # 将loss写入TensorBoard
                    tf.summary.scalar('ac_methond_auc_' + str(i), tmp_auc, step=methond_train_count)
                    summary_writer.flush()
            auc_str += str(methond_train_count)
            with open(eval_auc_file_path, 'a') as f:
                f.write(auc_str + '\n')
            print('Ac_Methond model validation AUC ============   ' + str(methond_auc) + '       steps ======== ' + str(
                methond_train_count))

            methond_early_flag = early_stop(methond_histy_va, 'loss', 0.0001, 80)
            if methond_early_flag:
                break

# Test the model
elif train_strategy == 5:
    ac_tr = []
    ac_va = []
    ac_te = []

    ac_tr_pre = get_data_test_set(ac_training_set, sp_read_and_decode, acm_batch_size, 1)
    ac_tr_it_pre = ac_tr_pre.as_numpy_iterator()
    ac_va_pre = get_data_test_set(ac_validation_set, sp_read_and_decode, acm_batch_size, 1)
    ac_va_it_pre = ac_va_pre.as_numpy_iterator()
    ac_te_pre = get_data_test_set(ac_test_set, sp_read_and_decode, acm_batch_size, 1)
    ac_te_it_pre = ac_te_pre.as_numpy_iterator()

    mci_te_pre = get_data_test_set(mci_test_set, sp_read_and_decode, acm_batch_size, 1)
    mci_te_it_pre = mci_te_pre.as_numpy_iterator()

    smc_te_pre = get_data_test_set(smc_test_set, acm_read_and_decode, acm_batch_size)
    smc_te_it_pre = smc_te_pre.as_numpy_iterator()

    model = keras.models.load_model(
        model_save_path + 'save_models/ac_210v_open_set_evidence_methond_adcn_mcismc_0.0005_32_2020v.h5',
        custom_objects=_custom_objects)
    # model.summary()
    sub_model = model
    # Model(inputs=model.input, outputs=model.output)
    sub_model.summary()
    tf.keras.backend.set_learning_phase(False)
    count = 0

    for data, ad, cn, rid, viscode in ac_te_it_pre:
        data = tf.reshape(data, [-1, steps, 2090])
        batch, x, y = data.shape
        # data = pad_sequences(data, maxlen=steps, dtype='float', value=-4.0)
        methond_data = methond_simplified(data.numpy()[:, :, 0:60])
        # data_va = pad_sequences(data_va, maxlen=steps, dtype='float', value=-4.0)
        # methond_and_pred_va = get_methond_batch(rid, viscode, methond_data,model_save_path + 'save_models/acm_0.0005_32_2v_test', dic_type=1,rid_is_array=True)
        methond_and_pred_va = get_methond_batch(rid, viscode, methond_data,
                                                model_save_path + 'save_models/ac_0.0005_32_210v_open_set_evidence_test_adcn_mcismc_2v',
                                                dic_type=1, rid_is_array=True)
        # methond_and_pred_va = get_methond_batch(rid, viscode, methond_data, model_save_path + 'save_models/ac_0.0005_32_240v_y4_validation',dic_type=1,rid_is_array=True)

        predict_result = sub_model.predict_on_batch(
            [data, methond_and_pred_va[12], methond_and_pred_va[13], methond_and_pred_va[14],
             methond_and_pred_va[0],
             methond_and_pred_va[1],
             methond_and_pred_va[2],
             methond_and_pred_va[3],
             methond_and_pred_va[4],
             methond_and_pred_va[5],
             methond_and_pred_va[6],
             methond_and_pred_va[7],
             methond_and_pred_va[8],
             methond_and_pred_va[9],
             methond_and_pred_va[10],
             methond_and_pred_va[11]])
        # predict_result = sub_model.predict([data, ad, cn, mci])
        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid, viscode, methond_data, ad, cn, np.zeros(batch), np.zeros(batch), methond_and_pred_va[12],
                      methond_and_pred_va[13], methond_and_pred_va[14]]
        tmp_reslut.extend(predict_result[1])
        # print(len(predict_result))
        # print(predict_result[0])
        # print(predict_result[0:12])
        # sys.exit(0)
        ac_va.append(tmp_reslut)

    for data, ad, cn, rid, viscode in mci_te_it_pre:
        # break
        data = tf.reshape(data, [-1, steps, 2090])
        batch, x, y = data.shape
        # data = pad_sequences(data, maxlen=steps, dtype='float', value=-4.0)
        methond_data = methond_simplified(data.numpy()[:, :, 0:60])
        # data_va = pad_sequences(data_va, maxlen=steps, dtype='float', value=-4.0)
        # methond_and_pred_va = get_methond_batch(rid, viscode, methond_data,model_save_path + 'save_models/acm_0.0005_32_2v_test', dic_type=1,rid_is_array=True)
        methond_and_pred_va = get_methond_batch(rid, viscode, methond_data,
                                                model_save_path + 'save_models/ac_0.0005_32_210v_open_set_evidence_test_adcn_mcismc_2v',
                                                dic_type=1,
                                                rid_is_array=True)
        predict_result = sub_model.predict_on_batch(
            [data, methond_and_pred_va[12], methond_and_pred_va[13], methond_and_pred_va[14],
             methond_and_pred_va[0],
             methond_and_pred_va[1],
             methond_and_pred_va[2],
             methond_and_pred_va[3],
             methond_and_pred_va[4],
             methond_and_pred_va[5],
             methond_and_pred_va[6],
             methond_and_pred_va[7],
             methond_and_pred_va[8],
             methond_and_pred_va[9],
             methond_and_pred_va[10],
             methond_and_pred_va[11]])
        # predict_result = sub_model.predict([data, ad, cn, mci])
        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid, viscode, methond_data, ad, cn, np.ones(batch), np.zeros(batch), methond_and_pred_va[12],
                      methond_and_pred_va[13], methond_and_pred_va[14]]
        tmp_reslut.extend(predict_result[1])
        # print(len(predict_result))
        # print(predict_result[0])
        # print(predict_result[0:12])
        # sys.exit(0)
        ac_va.append(tmp_reslut)

    for data, ad, cn, mci, rid, viscode in smc_te_it_pre:
        # break
        data = tf.reshape(data, [-1, steps, 2090])
        batch, x, y = data.shape
        # data = pad_sequences(data, maxlen=steps, dtype='float', value=-4.0)
        methond_data = methond_simplified(data.numpy()[:, :, 0:60])
        # data_va = pad_sequences(data_va, maxlen=steps, dtype='float', value=-4.0)
        # methond_and_pred_va = get_methond_batch(rid, viscode, methond_data,model_save_path + 'save_models/acm_0.0005_32_2v_test', dic_type=1,rid_is_array=True)
        methond_and_pred_va = get_methond_batch(rid, viscode, methond_data,
                                                model_save_path + 'save_models/ac_0.0005_32_210v_open_set_evidence_test_adcn_mcismc_2v',
                                                dic_type=1,
                                                rid_is_array=True)
        predict_result = sub_model.predict_on_batch(
            [data, methond_and_pred_va[12], methond_and_pred_va[13], methond_and_pred_va[14],
             methond_and_pred_va[0],
             methond_and_pred_va[1],
             methond_and_pred_va[2],
             methond_and_pred_va[3],
             methond_and_pred_va[4],
             methond_and_pred_va[5],
             methond_and_pred_va[6],
             methond_and_pred_va[7],
             methond_and_pred_va[8],
             methond_and_pred_va[9],
             methond_and_pred_va[10],
             methond_and_pred_va[11]])
        # predict_result = sub_model.predict([data, ad, cn, mci])
        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid, viscode, methond_data, ad, cn, mci, np.ones(batch), methond_and_pred_va[12],
                      methond_and_pred_va[13], methond_and_pred_va[14]]
        tmp_reslut.extend(predict_result[1])
        # print(len(predict_result))
        # print(predict_result[0])
        # print(predict_result[0:12])
        # sys.exit(0)
        ac_va.append(tmp_reslut)

    result_va = concate_result_2v(ac_va)

    m_rid, m_viscode, m_methond, m_ys_ture, m_ys_pred, m_path = get_modify_prediction_4v(result_va[0], result_va[1],
                                                                                         result_va[2], result_va[3:7],
                                                                                         result_va[7:10],
                                                                                         result_va[10:22],
                                                                                         completeness=0.6,
                                                                                         pred_high=[0.95, 0.95, 0.8],
                                                                                         pred_low=[0.05, 0.05, 0.01],
                                                                                         methond_high=0.65,
                                                                                         methond_low=0.3,
                                                                                         iscontaincogforall=True)

    # ys_true, openmax_rate = [], retun_index = [0.9], num_thresholds = 200
    combine_m_ys_ture, combine_m_ys_pred = get_data_result(m_ys_ture, m_ys_pred, 3)

    print('methond '
          ':  ', get_methond_count(m_methond, combine_m_ys_ture, 3))
    print('Trans count:  ', trans_methond_count)
    get_known2unknown_2v(m_rid, m_viscode, m_methond, combine_m_ys_ture, combine_m_ys_pred)

    # print('################',get_auc_2v(combine_m_ys_ture, openmax_rate=combine_m_ys_pred, retun_index=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95,0.98],
    #                                    rids=m_rid,viscodes=m_viscode,methonds=m_methond))
    # print('################',get_auc_2v(m_ys_ture, openmax_rate=m_ys_pred, retun_index=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]))
    # reslute_save_path = model_save_path + 'save_models/ac_0.0005_32_210v_open_set_evidence_adcn_mcismc',

    fpr, tpr, thresholds = roc_curve(combine_m_ys_ture[0],
                                     get_one_hot_index(combine_m_ys_pred, 0))  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
    roc_auc = auc(fpr, tpr)  # 求auc面积

    fpr_cn, tpr_cn, thresholds_cn = roc_curve(combine_m_ys_ture[1],
                                              get_one_hot_index(combine_m_ys_pred, 1))  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
    roc_auc_cn = auc(fpr_cn, tpr_cn)  # 求auc面积

    font1 = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 15,
             }
    font2 = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 10,
             }

    # np.save('/data/huangyunyou/result/closed_fpr.npy', fpr)
    # np.save('/data/huangyunyou/result/closed_tpr.npy', tpr)

    # fpr_2 = np.load('/data/huangyunyou/result/closed_MRI_fpr.npy')
    # tpr_2 = np.load('/data/huangyunyou/result/closed_MRI_tpr.npy')

    roc_auc_cn = auc(fpr_cn, tpr_cn)  # 求auc面积

    acc, result = get_score_final(combine_m_ys_ture, combine_m_ys_pred)
    # plt.scatter(x, y)
    print(acc, result)

    # , Unkown sen={1:.4f}, Global acc={2:.4f
    # , Unkown sen={1:.4f}, Global acc={2:.4f}
    # ,result[2][0],acc
    plt.scatter([result[0][1]], [result[0][0]], s=50,
                label='AD operating point (Sensitivity={0:.4f})'.format(result[0][0]))  # ad
    plt.scatter([result[1][1]], [result[1][0]], s=50,
                label='CN operating point (Sensitivity={0:.4f})'.format(result[1][0]))  # cn
    plt.scatter([result[2][1]], [result[2][0]], s=50,
                label='Unkown operating point (Sensitivity={0:.4f})'.format(result[2][0]))  # cn

    # , markerfacecolor='red', markeredgecolor='gray'
    # , markerfacecolor='red', markeredgecolor='gray'
    plt.plot(fpr, tpr, linewidth=1, color="red", label='AD ROC (AUC = {0:.4f})'.format(roc_auc))  # 画出当前分割数据的ROC曲线
    plt.plot(fpr_cn, tpr_cn, linewidth=1, color="black",
             label='CN ROC (AUC = {0:.4f})'.format(roc_auc_cn))  # 画出当前分割数据的ROC曲线

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity', font1)
    plt.ylabel('Sensitivity', font1)  # 可以使用中文，但需要导入一些库即字体
    plt.title('Diagnosis with our AI system', fontsize=15, fontweight='bold')
    plt.legend(loc="lower right", prop=font2)
    plt.show()
    pain(m_rid, m_viscode, m_methond, m_ys_ture)

