import os

import numpy as np

from gen_data.DauUtils import get_dau
from utils import model_conf
from utils.utils import shuffle_data

cov_name_list = ["NAC", "TKNC", "NBC", "SNAC"]
dau_name_arr = ['SF', 'ZM', 'BR', 'RT', 'CT', 'BL', 'SR']
mop_name_arr = ["IR", "ORI", "RG", "RP", "CR"]



# 消融实验
# rank_name_list = ["DeepDAC-var1", "DeepDAC-var2", "DeepDAC-var3"]

rank_name_list = ["DeepDAC"]


import cv2


def mk_exp_dir(exp_name, data_name, model_name, base_path):
    # 6.进行试验
    ## 6.1 创建文件夹并储存该次参数文件
    pair_name = model_conf.get_pair_name(data_name, model_name)  # return data_name + "_" + model_name
    dir_name = exp_name + "_" + pair_name
    base_path = base_path + "/" + dir_name
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    return base_path


def _get_IR_dau(data_name):
    if data_name == model_conf.mnist:
        dau = get_dau(model_conf.fashion)
    elif data_name == model_conf.fashion:
        dau = get_dau(model_conf.mnist)
    elif data_name == model_conf.cifar10:
        dau = get_dau(model_conf.svhn)
    elif data_name == model_conf.svhn:
        dau = get_dau(model_conf.cifar10)
    elif data_name == model_conf.cifar100:
        dau = get_dau(model_conf.svhn)
    elif data_name == model_conf.caltech:
        dau = get_dau(model_conf.cifar100)
    else:
        raise ValueError("Get IR dau wrongly")
    return dau


# 获取污染后的数据
def get_mutation_data(op, mop, x_dau, y_dau, data_name, seed=0):
    ratio = 0.2
    print("x_dau", len(x_dau), len(x_dau[0]))

    if op == "LB":
        x_arr = []
        y_arr = []
        for x_dop, y_dop in zip(x_dau, y_dau):
            x_lb, y_lb = mop.label_bias(x_dop, y_dop, seed=seed)  # LB 标签不均衡
            x_arr.append(x_lb)
            y_arr.append(y_lb)
        x_select = np.concatenate(x_arr, axis=0)
        y_select = np.concatenate(y_arr, axis=0)
    else:
        x_select = np.concatenate(x_dau, axis=0)
        y_select = np.concatenate(y_dau, axis=0)
        if op == "RG":  # RG 随机生成  随机生成的图片数据
            img_shape = x_select[0].shape
            x_add, y_add = mop.synthetic_data(len(x_select), img_shape, ratio=ratio, seed=seed)
        elif op == "RP":  # RP 随机挑选重复数据
            x_add, y_add = mop.repeat_data(x_select, y_select, ratio=ratio, seed=seed)
        elif op == "IR":  # 不相关数据
            dau = _get_IR_dau(data_name)
            (x_extra, _), (_, _) = dau.load_data(use_norm=True)
            if data_name == model_conf.caltech:
                x_extra = np.array([cv2.resize(i, (200, 200)) for i in x_extra])
            x_add, y_add = mop.irrelevant_data(len(x_select), x_extra, ratio=ratio, seed=seed)
            del _
        elif op == "CR":  # 数据破损
            x_add, y_add = mop.corruption_data(x_select, y_select, ratio=ratio, seed=seed)
        elif op == "NR":  # 数据噪音
            x_add, y_add = mop.noise_data(x_select, y_select, ratio=ratio, seed=seed)
        elif op == "PS":  # 数据投毒
            x_add, y_add = mop.poison_data(x_select, y_select, ratio=ratio, seed=seed)
        else:
            raise ValueError()
        print(op, "add", len(x_add))
        x_select = np.concatenate([x_select, x_add], axis=0)
        y_select = np.concatenate([y_select, y_add], axis=0)
    x_select, y_select = shuffle_data(x_select, y_select, 0)
    assert len(x_select) == len(y_select)
    return x_select, y_select


# 返回带扩增标签的数组
def get_dau_data(x_test, y_test, dau, dau_name_arr, ratio=0.5, shuffle=False):
    x_test_arr = []
    y_test_arr = []

    x_val_dict = {}
    y_val_dict = {}
    # 添加原始的
    num = int(len(x_test) * ratio)

    # 切分原始测试集合
    x_test_arr.append(x_test[:num])
    y_test_arr.append(y_test[:num])
    x_val_dict["ORI"] = x_test[num:]
    y_val_dict["ORI"] = y_test[num:]

    # 添加扩增的
    for dau_op_name in dau_name_arr:
        # print(dau_op_name)
        x, y = dau.load_dau_data(dau_op_name, use_norm=True, use_cache=False)
        if shuffle:
            x, y = shuffle_data(x, y, 0)
        num = int(len(x) * ratio)
        x_test_arr.append(x[:num])
        y_test_arr.append(y[:num])

        x_val_dict[dau_op_name] = x[num:]
        y_val_dict[dau_op_name] = y[num:]

    # x_dau_test = np.concatenate(x_test_arr, axis=0)
    # y_dau_test = np.concatenate(y_test_arr, axis=0)

    return x_test_arr, y_test_arr, x_val_dict, y_val_dict
