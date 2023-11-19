import os

from keras.backend import clear_session
from keras.engine.saving import load_model
from tqdm import tqdm

import exp_utils
from exp_utils import mk_exp_dir, get_dau_data
from gen_data.DauUtils import get_dau
from mop.mutation_op import Mop
from utils import model_conf
import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend('agg')


def exec(model_name, data_name, rank_type):
    # 实验 只选取了seed=0的作为统计数据
    if rank_type == "rank":
        rank_name_list = exp_utils.rank_name_list
        base_dir = "result_raw/Div{}".format("")
        suffix = ""
    elif rank_type == "cov":
        rank_name_list = exp_utils.cov_name_list
        base_dir = "result_raw/Cov{}".format("")
        suffix = "_EXT"
    else:
        raise ValueError()
    base_path = mk_exp_dir(exp_name, data_name, model_name, base_dir)
    exp_repeat(model_name, data_name, base_path, rank_name_list, suffix=suffix)


# 重复数据实验
def exp_repeat(model_name, data_name, base_path, rank_name_list, suffix=""):
    dau_name_arr = exp_utils.dau_name_arr  # 扩增方法
    mop_name_arr = exp_utils.mop_name_arr  # 变异方法
    print(dau_name_arr)
    dau = get_dau(data_name)
    mop = Mop()
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
    del x_train
    ps_path = "{}/ps_data/".format(base_path)
    diverse_error_dir = "{}/diverse_error/".format(base_path)
    fault_dir = "{}/fault/".format(base_path)
    time_path = "{}/time/".format(base_path)
    os.makedirs(time_path, exist_ok=True) # 排序时间
    os.makedirs(ps_path, exist_ok=True) # 排序的原始数据
    os.makedirs(diverse_error_dir, exist_ok=True) # 失效多样性
    os.makedirs(fault_dir, exist_ok=True) # 失效测试用例个数
    select_size_ratio_arr = [0.025, 0.05, 0.075, 0.1]
    split_ratio = 0.5
    sample_num = 1
    model_path = model_conf.get_model_path(data_name, model_name)
    print(model_path)
    #ori_model = load_model(model_path)
    ############################
    # 实验
    ############################

    # 构造原始选择集
    x_dau_test_arr, y_dau_test_arr, x_val_dict, y_val_dict = get_dau_data(x_test, y_test, dau, dau_name_arr,
                                                                          ratio=split_ratio,
                                                                          shuffle=True)

    select_size_arr = []
    len_data = len(np.concatenate(y_dau_test_arr, axis=0))
    for select_size_ratio in select_size_ratio_arr:
        select_size = int(len_data * select_size_ratio)
        select_size_arr.append(select_size)
    max_select_size = select_size_arr[-1]
    print("max_select_size", max_select_size)  # 最大测试用例个数
    for mop_name in mop_name_arr:
        ori_model = load_model(model_path)
        for i in range(sample_num):  # 每个算子5次,每次的数据组成都不一样
            if mop_name == "ORI":
                x_select = np.concatenate(x_dau_test_arr, axis=0)
                y_select = np.concatenate(y_dau_test_arr, axis=0)
            else:
                x_select, y_select = exp_utils.get_mutation_data(mop_name, mop, x_dau_test_arr, y_dau_test_arr,
                                                                 data_name, seed=i)
            '''
            每个噪音污染的方式都回得到对应的Fault number 和 Fault diversity
            '''
            save_path = os.path.join(ps_path, "{}_" + "{}_{}.npy".format(mop_name, i))
            diverse_error_path = os.path.join(diverse_error_dir, "{}_" + "{}_{}.npy".format(mop_name, i))
            fault_path = os.path.join(fault_dir, "{}_" + "{}_{}.npy".format(mop_name, i))
            # 优先级排序
            print("len(x_select)", len(x_select))

            '''
            获取每种方法的排序结果
            '''
            idx_data = {}
            for name in rank_name_list:
                fp_extra = save_path.format(name + suffix)
                if os.path.exists(fp_extra):
                    ix = np.load(fp_extra)
                else:
                    raise ValueError()
                assert len(ix) >= max_select_size
                idx_data[name] = ix
            print("mop_name :{}".format(mop_name), "keys:", idx_data.keys(), "select_size_arr:", select_size_arr)
            '''
            针对某种污染手段，获得每种排序方法的error_diverse以及fault_idx
            '''
            for k, idx in tqdm(idx_data.items()): # k:rank_name; idx: rank_indexs
                diverse_error_idx_path = diverse_error_path.format(k) # 各自的方法为后缀
                fault_idx_path = fault_path.format(k)
                x_s, y_s = x_select[idx][:max_select_size], y_select[idx][:max_select_size] # 选取的是最大数量的 slection_size
                save_diverse_error_idx(diverse_error_idx_path, ori_model, x_s, y_s)   # error diversity
                save_fault_idx_path(fault_idx_path, ori_model, x_s, y_s, mop_name)  # fault

            total_diverse_error_idx_path = os.path.join(diverse_error_dir, "_" + "{}_{}.npy".format(mop_name, i)) # 总的情况
            total_fault_idx_path = os.path.join(fault_dir, "_" + "{}_{}.npy".format(mop_name, i))
            print(f"total_diverse_error_idx_path is {total_diverse_error_idx_path} and total_fault_idx_path is {total_fault_idx_path}")
            save_diverse_error_idx(total_diverse_error_idx_path, ori_model, x_select, y_select) # 错误的diversity ground truth -> predicted 不同的多样性标签不同的
            print("Total_save_diverse_error_idx is completed!!!")
            save_fault_idx_path(total_fault_idx_path, ori_model, x_select, y_select, mop_name) # 样本中的所有失效索引
            print("Total_save_fault_idx_path is completed!!!")
            del x_select


def save_diverse_error_idx(error_idx_path, ori_model, x_s, y_s):
    '''
    获取的是错误样本的标签失效的多样性， 例如1->2 1->3 3->4
    '''
    y_prob = ori_model.predict(x_s)  # 预测概率
    del x_s
    y_psedu = np.argmax(y_prob, axis=1) # 根据概率得到预测标签
    fault_pair_arr = []
    fault_idx_arr = []
    for ix, (y_s_temp, y_psedu_temp) in enumerate(zip(y_s, y_psedu)):
        if y_s_temp == -1:
            continue
        elif y_s_temp == y_psedu_temp:
            continue
        else:
            key = (y_s_temp, y_psedu_temp)
            if key not in fault_pair_arr:  # 判断是否在fault_pair_array中, 如果是重复的也不会添加
                fault_pair_arr.append(key)
                fault_idx_arr.append(ix)  # 存储错误的索引
    np.save(error_idx_path, fault_idx_arr)


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
        elif y_s_temp == y_psedu_temp: # 正确样本
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
    exp_name = "exp_mutation"  # exp_repeat
    only_add = False

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    for data_name, v_arr in tqdm(model_conf.model_data.items()):
        for model_name in v_arr:
            print(model_name, data_name)
            #for rank_type in ["cov","rank"]:
            for rank_type in ["rank"]:
                exec(model_name, data_name, rank_type=rank_type)
