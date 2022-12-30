import os

from keras.datasets import mnist

from gen_data.Dau import Dau
import numpy as np
from keras.models import load_model



class MnistDau(Dau):
    def __init__(self):
        super().__init__("mnist")
        self.train_size = 60000  # 写死了
        self.test_size = 10000
        self.nb_classes = 10
        self.dataname='mnist'

    def minist_load_data(self, path):
        f = np.load(path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

    # 用于扩增图片
    # 加载原始数据
    def load_data(self, use_norm=False):  # use_norm=False 用于图片扩增  # use_norm=True 用于训练数据和实验
        (x_train, y_train), (x_test, y_test) = self.minist_load_data(path='./data/mnist.npz')
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
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
            "SF": [(0, 0.15), (0, 0.15)],
            "RT": (0, 40),  # rotation
            "ZM": ((0.5, 2.0), (0.5, 2.0)),  # zoom
            "BR": 0.5,
            "SR": [10, 40.0],  # sheer
            "BL": "hard",  # blur
            "CT": [0.5, 1.5],
        }
        print(params)
        return params


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dau = MnistDau()
    # dau.run("test")

    #############################################################################################################
    '''
    下面的注释主要是为了评估 在该数据集下的模型在各自增强方式下的精度
    '''
    # dau.show_test(dau.dataname)

    model_path = '../model/model_mnist_LeNet5.hdf5' # 返回模型的路径
    model =  load_model(model_path)
    dau.get_acc_by_op(os.path.abspath(model_path))
    model_path = "../model/model_mnist_LeNet1.hdf5"
    dau.get_acc_by_op(model_path)
