#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Q1mi"
# Date: 2022/6/17
import math
import os

import sklearn.model_selection
from keras.datasets import cifar100

from gen_data.Dau import Dau
import numpy as np

class CaltechDau(Dau):
    def __init__(self):
        super().__init__("caltech",nb_classes=101)
        self.train_size = 6674  # 写死了
        self.test_size = 1622
        self.nb_classes = 101

    def train_test_split(self, x, y, test_size=0.2, seed = 2022):

        from collections import defaultdict
        grouped_data = defaultdict(list)
        for i in range(len(y)):
            grouped_data[y[i]].append(x[i])

        np.random.seed(seed)

        train_data = []
        test_data = []
        train_label = []
        test_label = []
        for label, data in grouped_data.items():
            np.random.shuffle(data)
            train_size = math.ceil(len(data) * (1-test_size))
            train_data.extend(data[:train_size])
            test_data.extend(data[train_size:])
            train_label.extend([label] * train_size)
            test_label.extend([label] * (len(data) - train_size))
        # 拼接train和test
        x_train = np.stack(train_data, axis=0)
        x_test = np.stack(test_data, axis=0)
        y_train = np.stack(train_label, axis=0)
        y_test = np.stack(test_label, axis=0)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        return x_train, x_test, y_train, y_test

    # 用于扩增图片
    # 加载原始数据
    def load_data(self, use_norm=False):  # use_norm=False 用于图片扩增  # use_norm=True 用于训练数据和实验
        # x.shape: (200, 200, 3)
        x = np.load('./data/x200.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        y = np.load('./data/y200.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        # 根据y的label 对数据x,y进行分组
        seed = 2022
        np.random.seed(seed)
        # import pandas as pd
        x_train, x_test, y_train, y_test = self.train_test_split(x, y, test_size=0.2, seed=seed)

        self.train_size = len(x_train)
        self.test_size = len(x_test)
        if use_norm:
            x_train = np.array(x_train, dtype='float32')
            x_test = np.array(x_test, dtype='float32')
            x_train /= 255
            x_test /= 255
        return (x_train, y_train), (x_test, y_test)

    def get_dau_params(self):
        params = {
            "SF": [(0.05, 0.07), (0.05, 0.07)],
            "RT": (5, 7),  # rotation
            "ZM": ((0.7, 1.0), (0.7, 1.0)),  #
            "BR": 0.3,  #
            "SR": [15, 17],  #
            "BL": "easy",  #
            "CT": [0.5, 1.5],
            "NS": None  # noise
        }
        return params


if __name__ == '__main__':
    dau = CaltechDau()

    (x_train, y_train), (x_test, y_test) = dau.load_data()
    # 获取每个类别的数据数量，打印类别最少的数量
    # from collections import Counter
    # print(Counter(y_train))


    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # dau.run("test")
    #
       # from utils import model_conf
    # model_path = '../model/model_caltech_resNet20.h5'
    # dau.get_acc_by_op(model_path,nb_classes=101)
    # model_path ='../model/model_caltech_resNet32.h5'
    # dau.get_acc_by_op(model_path,nb_classes=101)
