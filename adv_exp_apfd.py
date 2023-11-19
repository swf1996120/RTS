import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
from keras.backend import clear_session
from keras.engine.saving import load_model
from tqdm import tqdm

import exp_utils
from exp_retrain_cov import expand_rank_ix
from exp_utils import mk_exp_dir, get_dau_data
from gen_data.DauUtils import get_dau
from mop.mutation_op import Mop
from utils import model_conf
import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend('agg')




def exec(model_name, data_name):
    # 实验 只选取了seed=0的作为统计数据

    rank_name_list = ["DeepDiv", "DeepDAC", "LSA", "Kmeas_sample", "NAC", "msp_gini", "entrory_dropout_rank", "CLUE_sample", "MAX_P"]

    base_dir = "result_raw/Adv"
    base_path = mk_exp_dir(exp_name, data_name, model_name, base_dir)
    print(base_path)
    exp_repeat(model_name, data_name, base_path, rank_name_list)


def apfd(right, sort):
    length = np.sum(sort != 0)
    if length != len(sort):
        sort[sort == 0] = np.random.permutation(len(sort) - length) + length + 1
    sum_all = np.sum(sort.values[[right.values != 1]])
    n = len(sort)
    m = pd.value_counts(right)[0]
    # print("apfd: sum_all = ", sum_all, "n = ", n, "m = ", m)
    return 1 - float(sum_all) / (n * m) + 1. / (2 * n)


# 重复数据实验
def exp_repeat(model_name, data_name, base_path, rank_name_list, suffix=""):
    dau = get_dau(data_name)

    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
    del x_train
    ps_path = "{}/ps_data/".format(base_path)

    os.makedirs(ps_path, exist_ok=True)  # 排序的原始数据

    model_path = model_conf.get_model_path(data_name, model_name)
    print(model_path)

    # ori_model = load_model(model_path)
    ############################
    # 实验
    ############################

    # 构造原始选择集
    attack_lst = ['fgsm', 'jsma', 'bim', 'cw']
    adv_image_all = []
    adv_label_all = []
    for attack in attack_lst:
        if data_name == model_conf.cifar10:
            adv_image_all.append(
                np.load('/data/swf/DAC/data/adv_image/{}_{}_{}_image.npy'.format(attack, "cifar10", model_name)))
            adv_label_all.append(
                np.load('/data/swf/DAC/data/adv_image/{}_{}_{}_label.npy'.format(attack, "cifar10", model_name)))
        else:
            adv_image_all.append(
                np.load('/data/swf/DAC/data/adv_image/{}_{}_{}_image.npy'.format(attack, data_name, model_name)))
            adv_label_all.append(
                np.load('/data/swf/DAC/data/adv_image/{}_{}_{}_label.npy'.format(attack, data_name, model_name)))
    adv_image_all = np.concatenate(adv_image_all, axis=0)
    adv_label_all = np.concatenate(adv_label_all, axis=0)
    x_select = np.concatenate([x_test, adv_image_all], axis=0)
    y_select = np.concatenate([y_test, adv_label_all], axis=0)

    save_path = os.path.join(ps_path, "{}.npy")

    # 优先级排序


    '''
    获取每种方法的排序结果
    '''
    max_select_size = len(y_select)
    idx_data = {}
    for name in rank_name_list:
        fp_extra = save_path.format(str(name) + suffix)
        if os.path.exists(fp_extra):
            if name in ["NAC", "NBC"]:
                fp = save_path.format(name)
                ix = np.load(fp, allow_pickle=True)
                ix = np.array(ix[2], dtype=np.int64)
                if len(ix) < max_select_size:
                    expand_ix_arr = np.random.permutation(len(x_select))
                    ix = expand_rank_ix(ix, expand_ix_arr, max_select_size)  # 每次就存储一个文件，保留最大的选择数目
            else:
                ix = np.load(fp_extra)
        else:
            raise ValueError()

        idx_data[name] = ix

    # for k, idx in tqdm(idx_data.items()):  # k:rank_name; idx: rank_indexs
    #     print(k, len(set(idx)))

    ori_model = load_model(model_path)


    # obtain predict label
    pred_test = np.argmax(ori_model.predict(x_select), axis=1)

    print("len(x_select)", len(x_select), "adv_image_all", len(adv_image_all), "incorrect", np.sum(pred_test != y_select))


    result = {}
    for k, idx in tqdm(idx_data.items()):  # k:rank_name; idx: rank_indexs
        df = pd.DataFrame([])
        df['right'] = (pred_test == y_select).astype('int')
        df['cam'] = 0
        df['cam'].loc[idx] = list(range(1, len(idx) + 1))
        df['rate'] = 0
        result[k] = apfd(df.right,df.cam)


    print("=====================================")
    print(data_name, model_name)
    for k, value in result.items():
        print(k +"\t"+str(value))
    print("=====================================")

def save_fault_idx_path(error_idx_path, ori_model, x_s, y_s, mop_name):
    '''
    @parameter: x_s: 候选集
    @parameter: y_s: 候选集对应的标签集
    获取的unique fault的样本的索引
    '''
    y_prob = ori_model.predict(x_s)
    y_psedu = np.argmax(y_prob, axis=1)
    fault_idx_arr = []
    for ix, (x_s_temp, y_s_temp, y_psedu_temp) in enumerate(zip(x_s, y_s, y_psedu)):
        if y_s_temp == -1:  # 噪音
            continue
        elif y_s_temp == y_psedu_temp:  # 正确样本
            continue
        else:
            if mop_name == "RP":  # 涉及到数据重复的问题
                add_flag = True
                for select_ix in fault_idx_arr:  # 之前选过的
                    if (x_s_temp == x_s[select_ix]).all():  # 数据重复
                        add_flag = False
                        break
                if add_flag:
                    fault_idx_arr.append(ix)
            else:
                fault_idx_arr.append(ix)
    np.save(error_idx_path, fault_idx_arr)


if __name__ == '__main__':
    exp_name = ""  # exp_repeat
    only_add = False


    from utils.model_conf import *

    model_data = {
        mnist: [LeNet5],
        fashion: [resNet20],
        cifar10: [vgg16],
        svhn: [vgg16],
        cifar100: [resNet32],
        caltech: [resNet20],
    }

    for data_name, v_arr in tqdm(model_data.items()):
        for model_name in v_arr:
            print(model_name, data_name)
            exec(model_name, data_name)
