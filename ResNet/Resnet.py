import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import copy
import random
from scipy.special import comb
import os
import matplotlib.image as image
from scipy.ndimage.interpolation import rotate
from skimage import io
from torchvision import transforms
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import shutil


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False)

def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=2):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=dilation, groups=groups, bias=False)

def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=3):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes ,batchNormalization, dropOut, stride=1, downsample=None, groups=1,
                 dilation=1, norm_layer=None ,num=1):
        super(BasicBlock, self).__init__()
        self.batchNormalization = batchNormalization
        self.dropOut = dropOut
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if num == 1:
            self.conv1 = conv7x7(inplanes, planes, stride)
            if self.batchNormalization == True:
                self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            if self.dropOut == True:
                self.do = nn.Dropout(p=0.3, inplace=False)
            self.conv2 = conv7x7(planes, planes)
            if self.batchNormalization == True:
                self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride
        elif num == 2:
            self.conv1 = conv5x5(inplanes, planes, stride)
            if self.batchNormalization == True:
                self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            if self.dropOut == True:
                self.do = nn.Dropout(p=0.3, inplace=False)
            self.conv2 = conv5x5(planes, planes)
            if self.batchNormalization == True:
                self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride
        elif num == 3:
            self.conv1 = conv3x3(inplanes, planes, stride)
            if self.batchNormalization == True:
                self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            if self.dropOut == True:
                self.do = nn.Dropout(p=0.3, inplace=False)
            self.conv2 = conv3x3(planes, planes)
            if self.batchNormalization == True:
                self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.batchNormalization == True:
            out = self.bn1(out)
        out = self.relu(out)
        if self.dropOut == True:
            out = self.do(out)

        out = self.conv2(out)
        if self.batchNormalization == True:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, batchNormalization, dropOut, num_classes=2, zero_init_residual=False,
                 groups=1, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.batchNormalization = batchNormalization
        self.dropOut = dropOut
        self.inplanes = 8
        self.dilation = 1

        self.groups = groups
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        if self.batchNormalization == True:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.dropOut == True:
            self.do = nn.Dropout(p=0.3, inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        downsample = nn.Sequential(
            conv1x1(8, 16, 1),
        )
        self.layer1 = block(8, 16, batchNormalization, dropOut, 1, downsample, self.groups, norm_layer, num=1)
        downsample1 = nn.Sequential(
            conv1x1(16, 32, 2),
        )
        self.layer2 = block(16, 32, batchNormalization, dropOut, 2, downsample1, self.groups, norm_layer, num=2)
        downsample3 = nn.Sequential(
            conv1x1(32, 64, 2),
        )
        self.layer3 = block(32, 64, batchNormalization, dropOut, 2, downsample3, self.groups, norm_layer, num=3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(64, 2),
                                nn.LogSoftmax(dim=1))

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        # print("1",x.shape)
        if self.batchNormalization == True :
            x = self.bn1(x)
        # print("2",x.shape)
        x = self.relu(x)
        if self.dropOut == True:
            x = self.do(x)
        # print("3",x.shape)
        x = self.maxpool(x)
        # print("4",x.shape)

        x = self.layer1(x)
        # print("5",x.shape)
        x = self.layer2(x)
        # print("6",x.shape)
        x = self.layer3(x)
        # print("7",x.shape)

        x = self.avgpool(x)
        # print("8",x.shape)
        x = torch.flatten(x, 1)
        # print("9",x.shape)
        x = self.fc(x)
        # print("10",x.shape)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet(batchNormalization, dropOut, **kwargs):
    model = ResNet(BasicBlock, [1, 1, 1], batchNormalization, dropOut, **kwargs)
    return model
