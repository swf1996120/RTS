import os

from cv2 import cv2

from selection_method.selection_utils import prepare_rank_ps

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

from tqdm import tqdm

import exp_utils
from exp_utils import mk_exp_dir
from gen_data.DauUtils import get_dau
from mop.mutation_op import Mop
from selection_method.nc_cover import get_cov_initer
from utils import model_conf
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需动态分配显存
session = tf.Session(config=config)
import matplotlib.pyplot as plt

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(0)
plt.switch_backend('agg')
from keras import backend as K
import numpy as np
from keras.engine.saving import load_model

'''
./exp_mutation_dataname_network/ps_data/TCPname_污染方式_0.npy:  测试用例排序算法的序列
'''


def get_fault_detection(model_name, data_name, seed):
    print(model_name, data_name)
    base_path = mk_exp_dir(exp_name, data_name, model_name, base_dir)
    exp(model_name, data_name, base_path, seed)
    K.clear_session()


def get_error(x_select, y_select, ori_model):
    x_pro = ori_model.predict(x_select)
    x_label = np.argmax(x_pro, axis=-1)
    count = 0
    for i in range(len(x_label)):
        if x_label[i] != y_select[i]:
            count += 1
    return count


# 重复数据实验
def exp(model_name, data_name, base_path, seed):
    model_path = model_conf.get_model_path(data_name, model_name)

    dau = get_dau(data_name)

    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)

    cov_initer = get_cov_initer(x_train, y_train, data_name, model_name)

    ############################
    # 实验
    ############################
    # 构造原始选择集

    correct_x, correct_y = np.load(os.path.join(base_path, 'correct_x.npy')), np.load(
        os.path.join(base_path, 'correct_y.npy'))
    failure_x, failure_y = np.load(os.path.join(base_path, 'failure_x.npy')), np.load(
        os.path.join(base_path, 'failure_y.npy'))
    noise_x, noise_y = np.load(os.path.join(base_path, 'noise_x.npy')), np.load(os.path.join(base_path, 'noise_y.npy'))

    normal_x = np.concatenate([correct_x, failure_x], axis=0)
    normal_y = np.concatenate([correct_y, failure_y], axis=0)

    sample_index = np.random.choice(range(len(noise_x)), size=int(len(normal_x) * 0.2), replace=False)

    noise_x = noise_x[sample_index]
    noise_y = noise_y[sample_index]

    normal_noise_x = np.concatenate([normal_x, noise_x], axis=0)
    normal_noise_y = np.concatenate([normal_y, noise_y], axis=0)

    print(normal_x.shape, normal_y.shape, normal_noise_x.shape, normal_noise_y.shape)
    select_size_ratio_arr = [0.025, 0.05, 0.075, 0.1, 0.2]

    select_size_arr = []
    len_data = len(normal_y)
    for select_size_ratio in select_size_ratio_arr:
        select_size = int(len_data * select_size_ratio)
        select_size_arr.append(select_size)
    max_select_size = select_size_arr[-1]
    print(f"The max_select_size is {max_select_size}")

    ori_model = load_model(model_path)

    print((np.argmax(ori_model.predict(correct_x), axis=-1) == correct_y).all(),
          (np.argmax(ori_model.predict(failure_x), axis=-1) != failure_y).all())

    rank_name_list = ["Gini","DeepDiv", ]

    ps_path = "{}/".format(base_path)

    nb_classes = model_conf.fig_nb_classes(data_name)

    data = {"normal": (normal_x, normal_y), "noise": (normal_noise_x, normal_noise_y)}

    for key, values in data.items():
        x_select, y_select = values

        save_path = os.path.join(ps_path, "{}_" + "{}.npy".format(key))

        df_ps = prepare_rank_ps(x_train, y_train, rank_name_list, model_path, x_select, save_path, cov_initer,
                                max_select_size, nb_classes, data_name)

    for key, values in data.items():
        print(f"Now is {key} type")
        x_select, y_select = values
        save_path = os.path.join(ps_path, "{}_" + "{}.npy".format(key))
        idx_data = {}
        for name in rank_name_list:
            fp = save_path.format(name)
            ix = np.load(fp)
            idx_data[name] = ix

        for k, idx in idx_data.items():
            name = str(k)
            print(f"{k}'s fault detecton")
            for cur_size_ratio, cur_select_size in zip(select_size_ratio_arr, select_size_arr):
                select_size = cur_select_size
                x_s, y_s = x_select[idx][:select_size], y_select[idx][:select_size]
                # 去除掉错误的数据
                x_s = x_s[y_s != -1]
                y_s = y_s[y_s != -1]

                error_number = get_error(x_s, y_s, ori_model)
                print(error_number / cur_select_size, end="\t")
            print("")


if __name__ == '__main__':
    from utils.model_conf import *

    '''
    '''
    base_dir = "motivation/"
    exp_name = "noise"
    model_data = {
        mnist: [LeNet5],
        svhn: [vgg16]
    }
    seed = 0
    for data_name, v_arr in tqdm(model_data.items()):
        for model_name in v_arr:
            get_fault_detection(model_name, data_name, seed)
