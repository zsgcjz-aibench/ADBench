from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import math
from torch.nn import Softmax
import torchvision.models as models
"""
design:

"""

class Model(nn.Module):
    def __init__(self, backbone, mlp, task):
        super(Model, self).__init__()
        self.backbone = backbone
        self.mlp = mlp
        self.softmax = nn.Softmax(dim=1) if task == 'ADD' else None

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp(x)
        if self.softmax: x = self.softmax(x)
        return x


class ECALayer(nn.Module):
    def __init__(self, channel):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


class MultiModal_CNN_Bone(nn.Module):
    def __init__(self, config, nonImg_size):
        super(MultiModal_CNN_Bone, self).__init__()
        self.nonImg_size = nonImg_size
        num, p = config['fil_num'], config['drop_rate']
        self.block1 = ConvLayer(1, num, (7, 2, 0), (3, 2, 0), p)
        self.block2 = ConvLayer(num, 2*num, (4, 1, 0), (2, 2, 0), p)
        self.se2 = SELayer(2*num, nonImg_size)
        self.block3 = ConvLayer(2*num, 4*num, (3, 1, 0), (2, 2, 0), p)
        self.se3 = SELayer(4*num, nonImg_size)
        self.block4 = ConvLayer(4*num, 8*num, (3, 1, 0), (2, 2, 0), p)
        self.se4 = SELayer(8*num, nonImg_size)
        self.size = self.test_size()

    def forward(self, x, feature):
        x = self.block1(x)
        x = self.block2(x)
        x = x + self.se2(x, feature)
        x = self.block3(x)
        x = x + self.se3(x, feature)
        x = self.block4(x)
        x = x + self.se4(x, feature)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x

    def test_size(self):
        case = torch.ones((2, 1, 182, 218, 182))
        feature = torch.ones((2, self.nonImg_size))
        output = self.forward(case, feature)
        return output.shape[1]


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pooling, drop_rate, BN=True, relu_type='leaky'):
        super(ConvLayer, self).__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type=='leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) 
       
    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class _CNN(nn.Module):
    def __init__(self, fil_num, drop_rate):
        super(_CNN, self).__init__()
        self.block1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block2 = ConvLayer(fil_num, 2 * fil_num, 0.1, (4, 1, 0), (2, 2, 0))
        self.block3 = ConvLayer(2 * fil_num, 4 * fil_num, 0.1, (3, 1, 0), (2, 2, 0))
        self.block4 = ConvLayer(4 * fil_num, 8 * fil_num, 0.1, (3, 1, 0), (2, 1, 0))
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(8 * fil_num * 6 * 8 * 6, 30),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(30, 2),
        )

    def forward(self, x, stage='normal'):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.dense1(x)
        if stage == 'get_features':
            return x
        else:
            x = self.dense2(x)
            return x


class ConvLayer2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pooling, drop_rate, BN=True, relu_type='leaky'):
        super(ConvLayer2, self).__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv1by1 = nn.Conv3d(in_channels, in_channels//2, 1, 1, 0)
        self.conv = nn.Conv3d(in_channels//2, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN1 = nn.BatchNorm3d(in_channels//2)
        self.BN2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type == 'leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv1by1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
class _FCN(nn.Module):
    def __init__(self, num, p):
        super(_FCN, self).__init__()
        self.features = nn.Sequential(
            # 47, 47, 47
            nn.Conv3d(1, num, 4, 1, 0, bias=False),
            nn.MaxPool3d(2, 1, 0),
            nn.BatchNorm3d(num),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # 43, 43, 43
            nn.Conv3d(num, 2*num, 4, 1, 0, bias=False),
            nn.MaxPool3d(2, 2, 0),
            nn.BatchNorm3d(2*num),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # 20, 20, 20
            nn.Conv3d(2*num, 4*num, 3, 1, 0, bias=False),
            nn.MaxPool3d(2, 2, 0),
            nn.BatchNorm3d(4*num),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # 9, 9, 9
            nn.Conv3d(4*num, 8*num, 3, 1, 0, bias=False),
            nn.MaxPool3d(2, 1, 0),
            nn.BatchNorm3d(8*num),
            nn.LeakyReLU(),
            # 6, 6, 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(8*num*6*6*6, 30),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(30, 2),
        )
        self.feature_length = 8*num*6*6*6
        self.num = num

    def forward(self, x, stage='train'):
        x = self.features(x)
        if stage != 'inference':
            x = x.view(-1, self.feature_length)
        x = self.classifier(x)
        return x

    def dense_to_conv(self):
        fcn = copy.deepcopy(self)
        A = fcn.classifier[1].weight.view(30, 8*self.num, 6, 6, 6)
        B = fcn.classifier[4].weight.view(2, 30, 1, 1, 1)
        C = fcn.classifier[1].bias
        D = fcn.classifier[4].bias
        fcn.classifier[1] = nn.Conv3d(160, 30, 6, 1, 0).cuda()
        fcn.classifier[4] = nn.Conv3d(30, 2, 1, 1, 0).cuda()
        fcn.classifier[1].weight = nn.Parameter(A)
        fcn.classifier[4].weight = nn.Parameter(B)
        fcn.classifier[1].bias = nn.Parameter(C)
        fcn.classifier[4].bias = nn.Parameter(D)
        return fcn


class _CNN_Bone(nn.Module):
    def __init__(self, config):
        super(_CNN_Bone, self).__init__()
        num, p = config['fil_num'], config['drop_rate']
        self.block1 = ConvLayer(1, num, (7, 2, 0), (3, 2, 0), p)
        self.block2 = ConvLayer(num, 2*num, (4, 1, 0), (2, 2, 0), p)
        self.block3 = ConvLayer(2*num, 4*num, (3, 1, 0), (2, 2, 0), p)
        self.block4 = ConvLayer(4*num, 8*num, (3, 1, 0), (2, 2, 0), p)
        self.size = self.test_size()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x

    def test_size(self):
        case = torch.ones((1, 1, 182, 218, 182))
        output = self.forward(case)
        return output.shape[1]


class MLP(nn.Module):
    def __init__(self, in_size, config):  # if binary out_size=2; trinary out_size=3
        super(MLP, self).__init__()
        fil_num, drop_rate, out_size = config['fil_num'], config['drop_rate'], config['out_size']
        self.fil_num = fil_num
        self.out_size = out_size
        self.in_size = in_size
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_size, fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num, out_size),
        )
        self.softmax = Softmax(dim=1)

    def forward(self, x, get_intermediate_score=False):
        x = self.dense1(x)
        if get_intermediate_score:
            return x
        x = self.dense2(x)
        x1 = self.softmax(x)
        return x1, x

    def dense_to_conv(self):
        fcn = copy.deepcopy(self)
        A = fcn.dense1[1].weight.view(self.fil_num, self.in_size//(6*6*6), 6, 6, 6)
        B = fcn.dense2[2].weight.view(self.out_size, self.fil_num, 1, 1, 1)
        C = fcn.dense1[1].bias
        D = fcn.dense2[2].bias
        fcn.dense1[1] = nn.Conv3d(self.in_size//(6*6*6), self.fil_num, 6, 1, 0).cuda()
        fcn.dense2[2] = nn.Conv3d(self.fil_num, self.out_size, 1, 1, 0).cuda()
        fcn.dense1[1].weight = nn.Parameter(A)
        fcn.dense2[2].weight = nn.Parameter(B)
        fcn.dense1[1].bias = nn.Parameter(C)
        fcn.dense2[2].bias = nn.Parameter(D)
        return fcn


class MLP2(nn.Module):
    def __init__(self, in_size, feature_size, config):  # if binary out_size=2; trinary out_size=3
        super(MLP2, self).__init__()
        fil_num, drop_rate, out_size = config['fil_num'], config['drop_rate'], config['out_size']
        self.fil_num = fil_num
        self.out_size = out_size
        self.in_size = in_size
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_size, fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num + feature_size, out_size),
        )

    def forward(self, x, y):
        x = self.dense1(x)
        x = torch.cat((x, y.float()), 1)
        x = self.dense2(x)
        return x


class MLP3(nn.Module):
    def __init__(self, mri_size, nonImg_size, config):
        super(MLP3, self).__init__()
        fil_num, drop_rate, out_size = config['fil_num'], config['drop_rate'], config['out_size']
        self.mri_emb = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(mri_size, fil_num),
            nn.BatchNorm1d(fil_num),
        )
        self.nonImg_emb = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(nonImg_size, fil_num),
            nn.BatchNorm1d(fil_num),
        )
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(2*fil_num, fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num, out_size),
        )

    def forward(self, mri, nonImg):
        mri = self.mri_emb(mri)
        nonImg = self.nonImg_emb(nonImg.float())
        x = torch.cat((mri, nonImg), 1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class LRP(nn.Module):
    """The model we use in the paper."""

    def __init__(self, dropout=0.4, dropout2=0.4):
        nn.Module.__init__(self)
        self.Conv_1 = nn.Conv3d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_1_mp = nn.MaxPool3d(2)
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_2_mp = nn.MaxPool3d(3)
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_3_mp = nn.MaxPool3d(2)
        self.Conv_4 = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.Conv_4_mp = nn.MaxPool3d(3)
        self.dense_1 = nn.Linear(2304, 32)
        self.dense_2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.dense_2(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class ResNet3D(nn.Module):
    def __init__(self, num_classes=2,
                 input_shape=(1, 110, 110, 110)):  # input: input_shape:	[num_of_filters, kernel_size] (e.g. [256, 25])
        super(ResNet3D, self).__init__()
        # stage 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=input_shape[0],
                out_channels=32,
                kernel_size=(3, 3, 3),
                padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3, 3),
                padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 3),
                stride=2,
                padding=1
            )
        )
        # stage 2
        self.bot2 = Bottleneck(64, 64, 1)
        # stage 3
        self.bot3 = Bottleneck(64, 64, 1)

        # stage 4
        self.conv4 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(
                in_channels=64,  # input height
                out_channels=64,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1,
                stride=2
            )
        )
        # stage 5
        self.bot5 = Bottleneck(64, 64, 1)
        # stage 6
        self.bot6 = Bottleneck(64, 64, 1)
        # stage 7
        self.conv7 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(
                in_channels=64,  # input height
                out_channels=128,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1,
                stride=2
            )
        )
        # stage 8
        self.bot8 = Bottleneck(128, 128, 1)

        # stage 9
        self.bot9 = Bottleneck(128, 128, 1)

        # stage 10
        self.conv10 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(7, 7, 7)))

        fc1_output_features = 128
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU()
        )

        fc2_output_features = 2
        self.fc2 = nn.Sequential(
            nn.Linear(fc1_output_features, fc2_output_features),
            nn.Sigmoid()
        )

    def forward(self, x, drop_prob=0.8):
        x = self.conv1(x)
        # print(x.shape)
        x = self.bot2(x)
        # print(x.shape)
        x = self.bot3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.bot5(x)
        # print(x.shape)
        x = self.bot6(x)
        # print(x.shape)
        x = self.conv7(x)
        # print(x.shape)
        x = self.bot8(x)
        # print(x.shape)
        x = self.bot9(x)
        # print(x.shape)
        x = self.conv10(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, num_filter * w * h)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        # prob = self.out(x) # probability
        return x


class Vox(nn.Module):
    def __init__(self, num_classes=2,
                 input_shape=(1, 110, 110, 110)):  # input: input_shape:	[num_of_filters, kernel_size] (e.g. [256, 25])
        super(Vox, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=input_shape[0],  # input height
                out_channels=8,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),

            nn.Conv3d(
                in_channels=8,  # input height
                out_channels=8,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),  # activation

            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,  # input height
                out_channels=16,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),  # activation
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),  # activation
            nn.Conv3d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=64,  # input height
                out_channels=64,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),  # activation
            nn.Conv3d(
                in_channels=64,  # input height
                out_channels=64,  # n_filters
                kernel_size=(3, 3, 3),  # filter size
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # choose max value in 2x2 area
        )

        fc1_output_features = 128
        self.fc1 = nn.Sequential(
            nn.Linear(100672, 512),  # 100672是182, 218, 182格式，13824是110,110,110格式
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        fc2_output_features = 64
        self.fc2 = nn.Sequential(
            nn.Linear(512, 250),
            nn.BatchNorm1d(250),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(250, fc2_output_features),
            nn.BatchNorm1d(fc2_output_features),
            nn.ReLU()
        )

        if (num_classes == 2):
            self.out = nn.Linear(fc2_output_features, 2)
            self.out_act = nn.Sigmoid()
        else:
            self.out = nn.Linear(fc2_output_features, num_classes)
            self.out_act = nn.Softmax()

    def forward(self, x, drop_prob=0.8):

        #print("1:", x.shape)  # torch.Size([8, 1, 110, 110, 110])，8个样本（batch_size大小），图像shape[1,110,110,110]
        x = self.conv1(x)
        # print("conv1:", x.shape)  # torch.Size([8, 8, 55, 55, 55])
        #		print(x.shape)
        x = self.conv2(x)
        # print("conv2",x.shape)  # torch.Size([8, 16, 27, 27, 27])
        #		print(x.shape)
        x = self.conv3(x)
        # print("conv3:", x.shape)  # torch.Size([8, 32, 13, 13, 13])
        #		print(x.shape)
        x = self.conv4(x)
        # print("conv4 x!!!!!!:", x.shape)  # torch.Size([8, 64, 6, 6, 6])  64*6*6*6=13824
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, num_filter * w * h)
        # x.size(0):8，调整成8行，-1：任意列
        # print("size xvvvvvvvvvvv:", x.shape)  # torch.Size([8, 13824])
        x = self.fc1(x)  # fc1中得第一个数：13824
        # print("x#####:", x.shape)  # torch.Size([8, 128])  # x的第一个数，fc1的第二个数
        x = nn.Dropout(drop_prob)(x)
        x = self.fc2(x)

        x = self.fc3(x)
        # x = nn.Dropout(drop_prob)(x)
        prob = self.out(x)  # probability
        y_hat = self.out_act(prob) # label
        # 		return y_hat, prob, x    # return x for visualization
        return y_hat

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
            nn.Linear(104544, 2000),  # 104544是（181, 217, 181)格式的.nii
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU()
        )

        if (num_classes == 2):
            self.out = nn.Linear(500, 2)
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
        prob = self.out(x)  # probability
        # 		y_hat = self.out_act(prob) # label
        # 		return y_hat, prob, x    # return x for visualization
        return prob


class att(nn.Module):
    def __init__(self, input_channel):
        "the soft attention module"
        super(att, self).__init__()
        self.channel_in = input_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=512,
                kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=64,
                kernel_size=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        mask = x
        mask = self.conv1(mask)
        mask = self.conv2(mask)
        mask = self.conv3(mask)
        att = self.conv4(mask)
        # print(att.size())
        output = torch.mul(x, att)
        return output

class _MLP_A(nn.Module):
    "MLP that only use DPMs from fcn"

    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP_A, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(fil_num)
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 2)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()

    def forward(self, X):
        X = self.bn1(X)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out

class Dynamic_images_VGG(nn.Module):
    def __init__(self,
                 num_classes=2,
                 feature='Vgg11',
                 feature_shape=(512, 7, 7),
                 pretrained=True,
                 requires_grad=True):

        super(Dynamic_images_VGG, self).__init__()

        # Feature Extraction
        if (feature == 'Alex'):
            self.ft_ext = models.alexnet(pretrained=pretrained)
            self.ft_ext_modules = list(list(self.ft_ext.children())[:-2][0][:9])

        elif (feature == 'Res34'):
            self.ft_ext = models.resnet34(pretrained=pretrained)
            self.ft_ext_modules = list(self.ft_ext.children())[0:3] + list(self.ft_ext.children())[4:-2]  # remove the Maxpooling layer
        elif (feature == 'Res18'):
            self.ft_ext = models.resnet18(pretrained=pretrained)
            self.ft_ext_modules = list(self.ft_ext.children())[0:3] + list(self.ft_ext.children())[
                                                                      4:-2]  # remove the Maxpooling layer

        elif (feature == 'Vgg16'):
            self.ft_ext = models.vgg16(pretrained=pretrained)
            self.ft_ext_modules = list(self.ft_ext.children())[0][:30]  # remove the Maxpooling layer

        elif (feature == 'Vgg11'):
            self.ft_ext = models.vgg11(pretrained=pretrained)
            self.ft_ext_modules = list(self.ft_ext.children())[0][:19]  # remove the Maxpooling layer

        elif (feature == 'Mobile'):
            self.ft_ext = models.mobilenet_v2(pretrained=pretrained)
            self.ft_ext_modules = list(self.ft_ext.children())[0]  # remove the Maxpooling layer

        self.ft_ext = nn.Sequential(*self.ft_ext_modules)
        for p in self.ft_ext.parameters():
            p.requires_grad = requires_grad

        # Classifier
        if (feature == 'Alex'):
            feature_shape = (256, 5, 5)
        elif (feature == 'Res34'):
            feature_shape = (512, 7, 7)
        elif (feature == 'Res18'):
            feature_shape = (512, 7, 7)
        elif (feature == 'Vgg16'):
            feature_shape = (512, 6, 6)
        elif (feature == 'Vgg11'):
            feature_shape = (512, 6, 6)
        elif (feature == 'Mobile'):
            feature_shape = (1280, 4, 4)

        conv1_output_features = int(feature_shape[0])
        print("conv1_output_features:", conv1_output_features)

        fc1_input_features = int(conv1_output_features * feature_shape[1] * feature_shape[2])
        fc1_output_features = int(conv1_output_features * 2)
        fc2_output_features = int(fc1_output_features / 4)

        self.attn = att(conv1_output_features)

        self.fc1 = nn.Sequential(
            nn.Linear(fc1_input_features, fc1_output_features),
            nn.BatchNorm1d(fc1_output_features),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(fc1_output_features, fc2_output_features),
            nn.BatchNorm1d(fc2_output_features),
            nn.ReLU()
        )

        self.out = nn.Linear(fc2_output_features, num_classes)

    def forward(self, x, drop_prob=0.5):
        x = self.ft_ext(x)
        # print(x.size())
        # print("1:", x.shape)
        x = self.attn(x)
        # print("2:", x.shape)
        # x = self.conv1(x)
        x = x.view(x.size(0), -1)
        # print("3:", x.shape)
        x = self.fc1(x)
        # print("4:", x.shape)
        x = nn.Dropout(drop_prob)(x)
        x = self.fc2(x)
        # print("5:", x.shape)
        x = nn.Dropout(drop_prob)(x)
        # print("6:", x.shape)
        prob = self.out(x)
        return prob


if __name__ == "__main__":
    pass

