#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Q1mi"
# Date: 2022/6/17

import os

from keras.datasets import cifar100

from gen_data.Dau import Dau


class Cifar100Dau(Dau):
    def __init__(self):
        super().__init__("cifar100",nb_classes=20)
        self.train_size = 50000  # 写死了
        self.test_size = 10000
        self.nb_classes = 20

    # 用于扩增图片
    # 加载原始数据
    def load_data(self, use_norm=False):  # use_norm=False 用于图片扩增  # use_norm=True 用于训练数据和实验
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)
        x_test = x_test.reshape(-1, 32, 32, 3)
        x_train = x_train.reshape(-1, 32, 32, 3)
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        if use_norm:
            x_test = x_test.astype('float32')
            x_train = x_train.astype('float32')
            x_train /= 255
            x_test /= 255
        return (x_train, y_train), (x_test, y_test)

    def get_dau_params(self):
        params = {
            "SF": [(0.05, 0.1), (0.05, 0.1)],
            "RT": (5, 10),  # rotation
            "ZM": ((0.7, 1.2), (0.7, 1.2)),  #
            "BR": 0.3,  #
            "SR": [15, 20],  #
            "BL": "easy",  #
            "CT": [0.5, 1.5],
            # "NS": "easy"  # noise
        }
        return params


if __name__ == '__main__':
    dau = Cifar100Dau()

    #
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # dau.run("test")
    #
    from utils import model_conf
    # model_path = '../model/model_cifar100_resNet20.h5'
    # dau.get_acc_by_op(model_path,nb_classes=20)

    # model_path = '../model/model_cifar100_resNet32.h5'
    # dau.get_acc_by_op(model_path,nb_classes=20)
