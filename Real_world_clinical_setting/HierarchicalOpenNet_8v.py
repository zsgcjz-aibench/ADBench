# -*- coding: utf-8 -*-

import logging
from functools import partial

import tensorflow.keras as keras
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
#from tensorflow.keras.layers.merge import _Merge
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Embedding, Lambda
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics.pairwise import paired_distances


class HierarchicalOpenNet_8v(object):
    """TadGAN model for time series reconstruction.

    Args:
        shape (tuple):
            Tuple denoting the shape of an input sample.
        encoder_input_shape (tuple):
            Shape of encoder input.
        generator_input_shape (tuple):
            Shape of generator input.
        critic_x_input_shape (tuple):
            Shape of critic_x input.
        critic_z_input_shape (tuple):
            Shape of critic_z input.
        layers_encoder (list):
            List containing layers of encoder.
        layers_generator (list):
            List containing layers of generator.
        layers_critic_x (list):
            List containing layers of critic_x.
        layers_critic_z (list):
            List containing layers of critic_z.
        optimizer (str):
            String denoting the keras optimizer.
        learning_rate (float):
            Optional. Float denoting the learning rate of the optimizer. Default 0.005.
        epochs (int):
            Optional. Integer denoting the number of epochs. Default 2000.
        latent_dim (int):
            Optional. Integer denoting dimension of latent space. Default 20.
        batch_size (int):
            Integer denoting the batch size. Default 64.
        iterations_critic (int):
            Optional. Integer denoting the number of critic training steps per one
            Generator/Encoder training step. Default 5.
        hyperparameters (dictionary):
            Optional. Dictionary containing any additional inputs.
    """

    def _build_model(self, model_type):

        layer1_num = self.layer1_num
        layer2_num = self.layer2_num
        layer3_num = self.layer3_num
        dropout_rate = self.dropout_rate
        steps = self.steps

        # region  #G_encode
        G_mask_layer = Masking(mask_value=-4.0, input_shape=(None, 2090*2), name='G_Bone_Mask')
        G_Lstm1 = layers.Bidirectional(
            layers.LSTM(layer1_num, return_sequences=True, dropout=dropout_rate, name='G_Bone_LSTM1'),
            name='G_Bone_BiLSTM1')
        G_nor1 = layers.BatchNormalization(name='G_Bone_Nor1')
        G_Lstm2 = layers.Bidirectional(
            layers.LSTM(layer2_num, return_sequences=True, dropout=dropout_rate, name='G_Bone_LSTM2'),
            name='G_Bone_BiLSTM2')
        G_nor2 = layers.BatchNormalization(name='G_Bone_Nor2')
        G_Lstm3 = layers.Bidirectional(layers.LSTM(layer3_num, dropout=dropout_rate, name='G_Bone_LSTM3'),
                                       name='G_Bone_BiLSTM3')
        G_nor3 = layers.BatchNormalization(name='G_Bone_Nor3')
        # endregion

        # region  #G_decode
        G_repeater = layers.RepeatVector(steps, name='G_Repeat')
        G_reverse_Lstm3 = layers.Bidirectional(
            layers.LSTM(layer3_num, return_sequences=True, dropout=dropout_rate, name='G_reverse_LSTM3'),
            name='G_reverse_BiLSTM3')
        # reverse_nor2 = layers.BatchNormalization(name='reverse_Nor2')
        G_reverse_Lstm2 = layers.Bidirectional(
            layers.LSTM(layer2_num, return_sequences=True, dropout=dropout_rate, name='G_reverse_LSTM2'),
            name='G_reverse_BiLSTM2')
        # reverse_nor1 = layers.BatchNormalization(name='reverse_Nor1')
        G_reverse_Lstm1 = layers.Bidirectional(
            layers.LSTM(layer1_num, return_sequences=True, dropout=dropout_rate, name='G_reverse_LSTM1'),
            name='G_reverse_BiLSTM1')
        G_TimeDis = layers.TimeDistributed(layers.Dense(2090, name='G_reverse_fc'), name='G_reverse_td')
        # endregion


        # region  #open ac cn 分类网络
        dense_ac1 = layers.Dense(32, activation='relu', name='ac_Dense1')
        drop_ac1 = layers.Dropout(dropout_rate, name='acm_Drop1')

        dense_ac2 = layers.Dense(16, activation='relu', name='ac_Dense2')
        # drop_ac1=layers.Dropout(dropout_rate,name='acm_Drop1')

        dense_ac3 = layers.Dense(8, activation='relu', name='ac_Dense3')
        # drop_ac1=layers.Dropout(dropout_rate,name='acm_Drop1')

        dense_ac4 = layers.Dense(2, activation='relu',name='ac_Dense4')

        output_layer = layers.Softmax(name='softmax_out')
        # endregion

        con_1 = layers.Concatenate(axis=1, name='concat_1')
        con_2 = layers.Concatenate(axis=1, name='concat_2')
        con_3 = layers.Concatenate(axis=1, name='concat_3')


        centers_layer = Embedding(2, 8+(2*layer1_num), name='embedding_label')

        if model_type=='AutoEncode_AC':
            inp_E = Input(shape=(None, 2090*2), name='inp_Encoder')
            input_label = Input(shape=(1), name='inp_label')

            x = G_mask_layer(inp_E)
            x = G_Lstm1(x)
            x = G_nor1(x)
            x = G_Lstm2(x)
            x = G_nor2(x)
            x = G_Lstm3(x)
            x = G_nor3(x)

            # 第一层
            ac_x_1n = dense_ac1(x)
            ac_x_1d = drop_ac1(ac_x_1n)
            x_r = G_repeater(x)
            x_r_1 = G_reverse_Lstm3(x_r)

            x_r_1_f = layers.GlobalAveragePooling1D()(x_r_1)
            x_r_1_f_ac = con_1([ac_x_1d, x_r_1_f])

            ac_x_1d_t = G_repeater(ac_x_1d)
            x_r_1_ac = tf.concat([x_r_1, ac_x_1d_t], axis=2)

            # 第二层
            ac_x_2n = dense_ac2(x_r_1_f_ac)
            x_r_2 = G_reverse_Lstm2(x_r_1_ac)

            x_r_2_f = layers.GlobalAveragePooling1D()(x_r_2)
            x_r_2_f_ac = con_2([ac_x_2n, x_r_2_f])

            ac_x_2n_t = G_repeater(ac_x_2n)
            x_r_2_ac = tf.concat([x_r_2, ac_x_2n_t], axis=2)

            # 第三层
            ac_x_3n = dense_ac3(x_r_2_f_ac)
            x_r_3 = G_reverse_Lstm1(x_r_2_ac)

            x_r_3_f = layers.GlobalAveragePooling1D()(x_r_3)
            x_r_3_f_ac = con_3([ac_x_3n, x_r_3_f])

            ac_x_3n_t = G_repeater(ac_x_3n)
            x_r_3_ac = tf.concat([x_r_3, ac_x_3n_t], axis=2)

            # 第四层
            ac_x_4n = dense_ac4(x_r_3_f_ac)
            out = output_layer(ac_x_4n)

            x_r = G_TimeDis(x_r_3_ac)

            centers = centers_layer(input_label)

            l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1]), 1, keepdims=True), name='l2_loss')(
                [x_r_3_f_ac, centers])

            return Model([inp_E,input_label], [out, x_r, l2_loss])


    def __init__(self,is_continue_train=False,save_path='',num_class=2):
        self.layer1_num = 96
        self.layer2_num = 64
        self.layer3_num = 32
        self.dropout_rate = 0.4
        self.steps = 127
        self.shape = (self.steps,2090*2) #输入数据形状
        self.latent_dim = self.layer3_num*2
        self.num_class=num_class
        #self.iterations_critic = iterations_critic
        #self.epochs = 20000
        #self.class_loss='M_BCE'

        self.optimizer =keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)

        self._build_tadgan(is_continue_train,save_path)

    def save_model(self,save_path):
        self.AutoEncode_AC.save(save_path+'_AutoEncode_AC_.h5')
        #self.encoder.save(save_path+'_encoder.h5')
        #self.generator.save(save_path+'_generator.h5')

    def center_loss(self,y_true,y_pred):
        return y_pred

    _custom_objects = {'center_loss': center_loss}

    def _load_model(self,save_path):
        #self.encoder_generator_model=keras.models.load_model(save_path + '_AutoEncode.h5')
        #self.critic_x_model=keras.models.load_model(save_path + '_CriticX.h5')
        #self.critic_z_model=keras.models.load_model(save_path + '_CriticZ.h5')
        #self.encoder=keras.models.load_model(save_path + '_encoder.h5')
        #self.generator=keras.models.load_model(save_path + '_generator.h5')
        self.AutoEncode_AC =keras.models.load_model(save_path + '_AutoEncode_AC_.h5',custom_objects=self._custom_objects)

    def compute_distance(a, b, metric_type='cosine'):
        return paired_distances(a, b, metric=metric_type, n_jobs=1)


    def _build_tadgan(self,is_continue_train=False,save_path=''):

        if not is_continue_train:
            #self.encoder = self._build_model('Encoder')
            #self.generator = self._build_model('Decoder')
            self.AutoEncode_AC = self._build_model('AutoEncode_AC')
        else:
            self._load_model(save_path)
        self.AutoEncode_AC.summary()


        self.AutoEncode_AC.compile(loss=['CategoricalCrossentropy', 'MSLE',self.center_loss],loss_weights=[0.49,0.26,0.25],optimizer=self.optimizer)

    def train_on_batch_customer(self,input, output, return_dict):
        return self.AutoEncode_AC.train_on_batch(input, output, return_dict=return_dict)

    def test_on_batch_customer(self,input, output, return_dict):
        return self.AutoEncode_AC.test_on_batch(input, output, return_dict=return_dict)


    def predict_customer(self, X):
        """Predict values using the initialized object.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.

        Returns:
            ndarray:
                N-dimensional array containing the reconstructions for each input sequence.
            ndarray:
                N-dimensional array containing the critic scores for each input sequence.
        """
        #X = X.reshape((-1, self.shape[0], 1))
        #z_ = self.encoder.predict(X)
        out = self.AutoEncode_AC.predict(X)

        middle_out_model = Model(inputs=self.AutoEncode_AC.input,
                                 outputs=[self.AutoEncode_AC.get_layer('G_Bone_BiLSTM3').output,
                                          self.AutoEncode_AC.get_layer('concat_1').output,
                                          self.AutoEncode_AC.get_layer('concat_2').output,
                                          self.AutoEncode_AC.get_layer('concat_3').output,
                                          self.AutoEncode_AC.get_layer('ac_Dense4').output]).predict(X)

        x_or=X[0][:,:,0:2090]
        loss = tf.keras.losses.mean_squared_logarithmic_error(tf.reshape(x_or, [x_or.shape[0], -1]), tf.reshape(out[1], [out[1].shape[0], -1]))
        #loss = tf.math.reduce_mean(tf.math.square(tf.reshape(x_or, [x_or.shape[0], -1]) - tf.reshape(out[1], [out[1].shape[0], -1])), axis=1)

        # print(loss)

        #print('&&&&&&&&&&&&&&&&&&&&&&  ', out[0])
        return middle_out_model[0], middle_out_model[1], middle_out_model[2], middle_out_model[3], middle_out_model[4], out[0], loss.numpy()

