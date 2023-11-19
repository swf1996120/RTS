import os

image = "image"
label = "label"
mnist = "mnist"
fashion = "fashion"
cifar10 = "cifar"
cifar100 = "cifar100"
caltech = "caltech"
svhn = "svhn"
resNet50 = "resNet50"
resNet32 = "resNet32"

LeNet5 = "LeNet5"
LeNet1 = "LeNet1"
resNet20 = "resNet20"
vgg16 = "vgg16"
MyLeNet5 = "MyLeNet5"
MyVgg16 = "MyVgg16"


def get_nb_classes(data_name):
    if data_name in [mnist, fashion, cifar10, svhn]:
        return 10
    elif data_name in [cifar100]:
        return 20
    elif data_name in [caltech]:
        return 101
    else:
        raise Exception("not in dataset")


fig_nb_classes = get_nb_classes

name_list = [mnist, fashion, svhn, cifar10, cifar100, caltech]
model_data = {
    mnist: [LeNet5, LeNet1],
    fashion: [LeNet1,resNet20],
    cifar10: [vgg16, resNet20],
    svhn: [LeNet5, vgg16],
    cifar100: [resNet20, resNet32],
    caltech: [resNet20,resNet32],
    # caltech: [resNet32]
}

pair_list = ["mnist_LeNet5", "mnist_LeNet1", "fashion_resNet20", "fashion_LeNet1", "svhn_LeNet5", "svhn_vgg16",
             "cifar_resNet20", "cifar_vgg16", "cifar100_resNet20", "cifar100_vgg16", "cifar100_resNet32",
             "caltech_vgg16", "caltech_resNet20"]


# 模型位置
def get_model_path(datasets, model_name):
    dic = {"mnist_LeNet5": './model/model_mnist_LeNet5.hdf5',
           "mnist_LeNet1": "./model/model_mnist_LeNet1.hdf5",
           "fashion_resNet20": "./model/model_fashion_resNet20.hdf5",
           "fashion_LeNet1": "./model/model_fashion_LeNet1.hdf5",
           "cifar_vgg16": "./model/model_cifar_vgg16.hdf5",
           "cifar_resNet20": "./model/model_cifar_resNet20.h5",
           "svhn_vgg16": "./model/model_svhn_vgg16.hdf5",
           "svhn_LeNet5": "./model/model_svhn_LeNet5.hdf5",
           "cifar100_resNet20": './model/model_cifar100_resNet20.h5',
           "cifar100_vgg16": './model/model_cifar100_vgg16.hdf5',
           "cifar100_resNet32": './model/model_cifar100_resNet32.h5',
           "caltech_vgg16": './model/model_caltech_vgg16.hdf5',
           "caltech_resNet20": './model/model_caltech_resNet20.h5',
           "caltech_resNet32": './model/model_caltech_resNet32.h5',
           }
    return dic[datasets + "_" + model_name]


def get_temp_model_path(datasets, model_name, smaple_method):
    path = './temp_model/' + datasets + "/" + model_name + "/" + smaple_method
    return path


def get_pair_name(data_name, model_name):
    return data_name + "_" + model_name
