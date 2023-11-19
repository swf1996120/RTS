import os

from keras.datasets import cifar10

from gen_data.Dau import Dau


class CifarDau(Dau):
    def __init__(self):
        super().__init__("cifar")
        self.train_size = 50000  # 写死了
        self.test_size = 10000
        self.nb_classes = 10

    # 用于扩增图片
    # 加载原始数据
    def load_data(self, use_norm=False):  # use_norm=False 用于图片扩增  # use_norm=True 用于训练数据和实验
        print("CIFAR dataset")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
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
            "SF": [(0, 0.15), (0, 0.15)],
            "RT": (5, 20),  # rotation
            "ZM": ((0.8, 1.5), (0.8, 1.5)),  # zoom
            "BR": 0.3,
            "SR": [10, 30],  # sheer
            "CT": [0.5, 1.5],
            "BL": None,  # blur
        }
        return params


if __name__ == '__main__':
    dau = CifarDau()

    #
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # dau.run("test")
    #
    from utils import model_conf
    model_path = "../model/model_cifar_resNet20.h5"
    dau.get_acc_by_op(model_path)
    model_path = "../model/model_cifar_vgg16.hdf5"
    dau.get_acc_by_op(model_path)
