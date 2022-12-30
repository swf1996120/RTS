#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Q1mi"
# Date: 2022/6/17

import os

from keras.datasets import cifar100

from gen_data.Dau import Dau
from sklearn.model_selection import train_test_split
import numpy as np

class CaltechDau(Dau):
    def __init__(self):
        super().__init__("caltech",nb_classes=101)
        self.train_size = 6636  # 写死了
        self.test_size = 1660
        self.nb_classes = 101

    # 用于扩增图片
    # 加载原始数据
    def load_data(self, use_norm=False):  # use_norm=False 用于图片扩增  # use_norm=True 用于训练数据和实验
        # x.shape: (200, 200, 3)
        x = np.load('./data/x200.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        y = np.load('./data/y200.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        seed = 2022
        np.random.seed(seed)
        # import pandas as pd
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        x_train = np.stack(x_train, axis=0)
        y_train = np.stack(y_train, axis=0)
        x_test = np.stack(x_test, axis=0)
        y_test = np.stack(y_test, axis=0)

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


    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dau.run("test")
    #
       # from utils import model_conf
    # model_path = '../model/model_caltech_vgg16.hdf5'
    # dau.get_acc_by_op(model_path,nb_classes=101)
    # model_path ='../model/model_caltech_resNet20.h5'
    # dau.get_acc_by_op(model_path,nb_classes=101)
