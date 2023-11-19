from gen_data.ImagenetDau import ImagenetDau
from utils import model_conf


def get_dau(data_name):
    '''
    各自的数据集加载方式
    '''
    from gen_data.CifarDau import CifarDau
    from gen_data.FashionDau import FashionDau
    from gen_data.MnistDau import MnistDau
    from gen_data.SvhnDau import SvhnDau
    from gen_data.Cifar100Dau import Cifar100Dau
    from gen_data.CaltechDau import CaltechDau
    if data_name == model_conf.mnist:
        return MnistDau()
    if data_name == model_conf.fashion:
        return FashionDau()
    if data_name == model_conf.svhn:
        return SvhnDau()
    if data_name == model_conf.cifar10:
        return CifarDau()
    if data_name == model_conf.cifar100:
        return Cifar100Dau()
    if data_name == model_conf.caltech:
        return CaltechDau()
    if data_name == model_conf.imagenet:
        return ImagenetDau()
    else:
        raise Exception("加载不到数据")



def get_data_size(data_name):
    '''
    返回各个数据集的训练集和测试集的大小
    '''
    if data_name == model_conf.mnist:
        train_size, test_size = 60000, 10000
    elif data_name == model_conf.fashion:
        train_size, test_size = 60000, 10000
    elif data_name == model_conf.svhn:
        train_size, test_size = 73257, 26032
    elif data_name == model_conf.cifar10:
        train_size, test_size = 50000, 10000
    elif data_name ==model_conf.cifar100:
        train_size, test_size = 50000, 10000
    elif data_name == model_conf.caltech:
        train_size, test_size = 6636, 1660
    elif data_name == model_conf.imagenet:
        train_size, test_size = 50000, 10000
    else:
        raise ValueError()
    return train_size, test_size
