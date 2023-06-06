import os
import pdb
import math
import pickle
import libmr
import joblib
import numpy as np
import tensorflow as tf
import HierarchicalOpenNet_8v
from sklearn.metrics.pairwise import paired_distances
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K

base_path = os.path.abspath(os.path.dirname(__file__))
result_data_path = os.path.join(base_path, 'result_data')
if not os.path.exists(result_data_path):
    os.mkdir(result_data_path)
steps = 127
acm_batch_size = 64
ac_training_set = '/home/user/ADNI/TFRecord_2v/ac_train.tfrecord'
ac_validation_set = '/home/user/ADNI/TFRecord_2v/ac_eval.tfrecord'
ac_test_set = '/home/user/ADNI/TFRecord_2v/ac_test.tfrecord'
mci_test_set = '/home/user/ADNI/TFRecord_2v/mci_test.tfrecord'


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


def get_singel_label(data_y, batch_len, class_num):
    ret_value = []
    for i in range(batch_len):
        for j in range(class_num):
            if data_y[j][i] == 1.0:
                ret_value.append(j)
                break
    return np.array(ret_value)


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


def weibull_fit_tails(av_map, tail_size=2000, metric_type='cosine'):
    weibull_model = {}
    labels = av_map.keys()

    for label in labels:
        print(f'EVT fitting for label {label}')
        weibull_model[label] = {}

        class_av = av_map[label]
        class_mav = np.mean(class_av, axis=0, keepdims=True)

        av_distance = np.zeros((1, class_av.shape[0]))
        for i in range(class_av.shape[0]):
            av_distance[0, i] = compute_distance(class_av[i, :].reshape(1, -1), class_mav, metric_type=metric_type)

        weibull_model[label]['mean_vec'] = class_mav
        weibull_model[label]['distances'] = av_distance

        mr = libmr.MR()

        tail_size_fix = min(tail_size, av_distance.shape[1])
        tails_to_fit = sorted(av_distance[0, :])[-tail_size_fix:]
        mr.fit_high(tails_to_fit, tail_size_fix)

        weibull_model[label]['weibull_model'] = mr

    return weibull_model


def compute_distance(a, b, metric_type='cosine'):
    return paired_distances(a, b, metric=metric_type, n_jobs=1)


def weibull_fit(model_path, score, prob, y):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    print('Model do not exists !      ', model_path)
    predicted_y = np.argmax(prob, axis=1)

    labels = np.unique(y)
    av_map = {}

    for label in labels:
        av_map[label] = score[(y == label) & (predicted_y == y), :]

    model = weibull_fit_tails(av_map, tail_size=300)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model


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
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                    elif index == 1:
                        a = total_acm / (2 * cn)
                        b = total_acm / (2 * (ad + mci))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                    elif index == 2:
                        a = total_acm / (2 * mci)
                        b = total_acm / (2 * (ad + cn))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                if self.model_type == 'sp':
                    if index == 0:
                        a = total_sp / (2 * smci)
                        b = total_sp / (2 * pmci)
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                    elif index == 1:
                        a = total_sp / (2 * pmci)
                        b = total_sp / (2 * smci)
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                if self.model_type == 'methond':

                    if index == 0:
                        a = x_total / (2 * x2)
                        b = x_total / (2 * (x_total - x2))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 1:
                        a = x_total / (2 * x3)
                        b = x_total / (2 * (x_total - x3))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 2:
                        a = x_total / (2 * x4)
                        b = x_total / (2 * (x_total - x4))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 3:
                        a = x_total / (2 * x5)
                        b = x_total / (2 * (x_total - x5))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 4:
                        a = x_total / (2 * x6)
                        b = x_total / (2 * (x_total - x6))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 5:
                        a = x_total / (2 * x7)
                        b = x_total / (2 * (x_total - x7))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 6:
                        a = x_total / (2 * x8)
                        b = x_total / (2 * (x_total - x8))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 7:
                        a = x_total / (2 * x9)
                        b = x_total / (2 * (x_total - x9))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 8:
                        a = x_total / (2 * x10)
                        b = x_total / (2 * (x_total - x10))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 9:
                        a = x_total / (2 * x11)
                        b = x_total / (2 * (x_total - x11))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 10:
                        a = x_total / (2 * x12)
                        b = x_total / (2 * (x_total - x12))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 11:
                        a = x_total / (2 * x13)
                        b = x_total / (2 * (x_total - x13))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                if self.model_type == 'ac_methond':

                    if index == 0:
                        a = ac_x_total / (2 * ac_x2)
                        b = ac_x_total / (2 * (ac_x_total - ac_x2))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 1:
                        a = ac_x_total / (2 * ac_x3)
                        b = ac_x_total / (2 * (ac_x_total - ac_x3))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 2:
                        a = ac_x_total / (2 * ac_x4)
                        b = ac_x_total / (2 * (ac_x_total - ac_x4))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 3:
                        a = ac_x_total / (2 * ac_x5)
                        b = ac_x_total / (2 * (ac_x_total - ac_x5))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 4:
                        a = ac_x_total / (2 * ac_x6)
                        b = ac_x_total / (2 * (ac_x_total - ac_x6))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 5:
                        a = ac_x_total / (2 * ac_x7)
                        b = ac_x_total / (2 * (ac_x_total - ac_x7))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 6:
                        a = ac_x_total / (2 * ac_x8)
                        b = ac_x_total / (2 * (ac_x_total - ac_x8))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 7:
                        a = ac_x_total / (2 * ac_x9)
                        b = ac_x_total / (2 * (ac_x_total - ac_x9))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 8:
                        a = ac_x_total / (2 * ac_x10)
                        b = ac_x_total / (2 * (ac_x_total - ac_x10))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 9:
                        a = ac_x_total / (2 * ac_x11)
                        b = ac_x_total / (2 * (ac_x_total - ac_x11))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 10:
                        a = ac_x_total / (2 * ac_x12)
                        b = ac_x_total / (2 * (ac_x_total - ac_x12))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
                                          0],
                                      -1)
                    elif index == 11:
                        a = ac_x_total / (2 * ac_x13)
                        b = ac_x_total / (2 * (ac_x_total - ac_x13))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 0.9999)) + b * (
                                    1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 0.9999)))) +
                                      log_var[
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


class CustomMultiLossLayer_AECS(layers.Layer):
    def __init__(self, nb_outputs=2, model_type='ac', task_weight=0.8, loss_type='cross_entropy',
                 name="CustomMultiLossLayer_AECS", **kwargs):
        self.nb_outputs = nb_outputs
        self.model_type = model_type
        self.loss_type = loss_type
        self.task_weight = task_weight
        self.is_placeholder = True
        super(CustomMultiLossLayer_AECS, self).__init__(name=name, **kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer_AECS, self).build(input_shape)

    def get_config(self):
        config = {"nb_outputs": self.nb_outputs, "model_type": self.model_type, "task_weight": self.task_weight,
                  "loss_type": self.loss_type}
        base_config = super(CustomMultiLossLayer_AECS, self).get_config()
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

                x2 = 52112
                x3 = 64616
                x4 = 61292
                x5 = 87406
                x6 = 76820
                x7 = 35056
                x8 = 10568
                x9 = 37199
                x10 = 9740
                x11 = 4605
                x12 = 8070
                x13 = 4452

                x_total = 338130

                if self.model_type == 'acm':
                    if index == 0:
                        a = total_acm / (2 * ad)
                        b = total_acm / (2 * (cn + mci))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                    elif index == 1:
                        a = total_acm / (2 * cn)
                        b = total_acm / (2 * (ad + mci))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                    elif index == 2:
                        a = total_acm / (2 * mci)
                        b = total_acm / (2 * (ad + cn))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                if self.model_type == 'ac':
                    if index == 0:
                        a = total_acm / (2 * ad)
                        b = total_acm / (2 * cn)
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                    elif index == 1:
                        a = total_acm / (2 * cn)
                        b = total_acm / (2 * ad)
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                if self.model_type == 'sp':
                    if index == 0:
                        a = total_sp / (2 * smci)
                        b = total_sp / (2 * pmci)
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                    elif index == 1:
                        a = total_sp / (2 * pmci)
                        b = total_sp / (2 * smci)
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0], -1)
                if self.model_type == 'methond':

                    if index == 0:
                        a = x_total / (2 * x2)
                        b = x_total / (2 * (x_total - x2))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 1:
                        a = x_total / (2 * x3)
                        b = x_total / (2 * (x_total - x3))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 2:
                        a = x_total / (2 * x4)
                        b = x_total / (2 * (x_total - x4))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 3:
                        a = x_total / (2 * x5)
                        b = x_total / (2 * (x_total - x5))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 4:
                        a = x_total / (2 * x6)
                        b = x_total / (2 * (x_total - x6))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 5:
                        a = x_total / (2 * x7)
                        b = x_total / (2 * (x_total - x7))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 6:
                        a = x_total / (2 * x8)
                        b = x_total / (2 * (x_total - x8))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 7:
                        a = x_total / (2 * x9)
                        b = x_total / (2 * (x_total - x9))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 8:
                        a = x_total / (2 * x10)
                        b = x_total / (2 * (x_total - x10))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 9:
                        a = x_total / (2 * x11)
                        b = x_total / (2 * (x_total - x11))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 10:
                        a = x_total / (2 * x12)
                        b = x_total / (2 * (x_total - x12))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)
                    elif index == 11:
                        a = x_total / (2 * x13)
                        b = x_total / (2 * (x_total - x13))
                        loss += K.sum(precision * (
                            -(a * y_true * tf.math.log(y_pred) + b * (1 - y_true) * tf.math.log(1 - y_pred))) + log_var[
                                          0],
                                      -1)

            elif self.loss_type == 'MSE':
                loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
            index += 1
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:2 * self.nb_outputs]
        loss = self.multi_loss(ys_true, ys_pred)

        ys_s_true = inputs[2 * self.nb_outputs]
        ys_s_pred = inputs[2 * self.nb_outputs + 1]
        loss_s = keras.losses.mean_squared_logarithmic_error(ys_s_true, ys_s_pred)
        # loss_s = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(ys_s_true-ys_s_pred)))

        self.add_loss(self.task_weight * loss + (1 - self.task_weight) * loss_s, inputs=inputs)
        # We won't actually use the output.
        # return ys_true,ys_pred
        return ys_true, ys_pred
        # K.concatenate(inputs, -1)


_custom_objects = {
    "CustomMultiLossLayer": CustomMultiLossLayer, "CustomMultiLossLayer_AECS": CustomMultiLossLayer_AECS,
}

if __name__ == '__main__':
    # region 载入数据
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
    # endregion

    # region 载入模型
    AE_model_pre = keras.models.load_model(
        os.path.abspath(os.path.join(base_path, 'model/dianli/ae_0.0005_32_22v.h5')),
        custom_objects=_custom_objects
    )

    AE_model = HierarchicalOpenNet_8v.HierarchicalOpenNet_8v(
        is_continue_train=True,
        save_path=os.path.abspath(os.path.join(base_path, 'model/dianli/ac_0.0005_64_700v')),
    )
    tf.keras.backend.set_learning_phase(False)
    # endregion

    # region 预测
    """
    data.shape: (64, 1, 265430), dtype: np.float32
    ad.shape: (64, 1), dtype: np.float32
    cn.shape: (64, 1), dtype: np.float32
    rid.shape: (64, 1), dtype: np.O（字节型）
    viscode.shape: (64, 1), dtype: np.O（字节型）
    """
    count = 0
    for data, ad, cn, rid, viscode in ac_tr_it_pre:

        count += 1
        batch, x, y = data.shape
        data = tf.reshape(data, [batch, -1, 2090])
        data_or = data
        data_g = AE_model_pre.predict_on_batch(data)
        data_g = tf.reshape(data_g, [batch, -1, 2090])
        data = tf.concat([data, data_g], axis=2)
        dummy_matrix = np.zeros((batch, 1))

        predict_result = AE_model.predict_customer([data, dummy_matrix])  # z,layer1,layer2,layer3,layer4,pred,loss
        # print([rid,predict_result[0][0]])
        tmp_reslut = [rid, viscode, methond_simplified(data.numpy()[:, :, 0:60])]
        tmp_reslut.append(predict_result[0])  # z                 3
        tmp_reslut.append(predict_result[1])  # 1                 4
        tmp_reslut.append(predict_result[2])  # 2                 5
        tmp_reslut.append(predict_result[3])  # 3                 6
        tmp_reslut.append(predict_result[4])  # 4                 7
        tmp_reslut.append(predict_result[5])  # pred(one-hot)     8
        tmp_reslut.append(predict_result[6])  # loss              9
        tmp_reslut.append(get_singel_label([ad, cn], batch, 2))  # 10
        # print('#####################################################')
        # print(tmp_reslut)
        ac_tr.append(tmp_reslut)

    joblib.dump(
        value=ac_tr,
        filename=os.path.abspath(os.path.join(base_path, 'result_data/GAN4_train_strategy_72_ac_tr.dat'))
    )

    pdb.set_trace()

    result_tr = concate_result_3v(ac_tr, muti_valu_index=8)

    joblib.dump(
        value=result_tr,
        filename=os.path.abspath(os.path.join(base_path, 'result_data/GAN4_train_strategy_72_result_tr.dat'))
    )

    print('Start train weibull model ..... ..... ..... ..... ..... ..... ..... ')
    # endregion

    # region 加入韦伯分布
    model = weibull_fit(
        os.path.abspath(os.path.join(base_path, 'model/save_model/ac_0.0005_32_700v_y4_weibull_model.pkl')),
        result_tr[7],
        result_tr[8],
        result_tr[10]
    )

    joblib.dump(
        value=model,
        filename=os.path.abspath(os.path.join(base_path, 'model/save_model/GAN4_train_strategy_72_weibull_model.dat'))
    )

    print('Complete train layer4_weibull model ..... ..... ..... ..... ..... ..... ..... ')
    # endregion
