import os
from tqdm import tqdm

import exp_utils
from exp_utils import mk_exp_dir, get_dau_data, get_mutation_data
from gen_data.DauUtils import get_dau
from mop.mutation_op import Mop
from selection_method.nc_cover import get_cov_initer
from selection_method.selection_utils import prepare_cov_ps
from utils.train_utils import retrain_detail, get_retrain_csv_data
from utils import model_conf
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = False  # 按需动态分配显存
session = tf.Session(config=config)

import matplotlib.pyplot as plt

plt.switch_backend('agg')
from utils.utils import add_df
from keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
'''
基本的思路如下：

注意的是：这里涉及到根据不同的模型去获得层的相关信息，这里已经封装好了，直接调用就可以；

1. 首先利用gen_data程序 去构造数据集中的每个样本的七种数据增强方法产生的样本；

2. 然后根据原始样本+七种数据增强的样本 --> 各自按照百分50%去进行划分成test(待排序的集合) ,  val集合 (模型验证性能的集合）

3. 对于待排序的集合 采用四种数据污染方法对数据进行污染，并分别统计在原始+四种污染数据上的测试用例选择的方法有效性

4. 对于选择方法需要注意的是， 对于污染的数据需要排除，只是起到扰乱的作用

5. 对于基于神经覆盖的方法： 
        ./exp_mutation_data_network/ps_data/TCPmethod_污染类型_0.npy 存储的是方法的神经覆盖情况
        ./exp_mutation_data_network/ps_data/TCPmethod_EXT_污染类型_0.npy 根据预设置的最大测试用例选择个数，存储的是方法的测试用例排序序列，不足的用随机填充    
        ./exp_mutation_data_network/time： 存储是各个排序方法所用的时间
        ./exp_mutation_data_network/res_污染类型_0.csv： 存储的各个方法在不同selection size下的训练精度差 以及其他的指标
'''


def exec(model_name, data_name, base_dir, seed):
    # 实验
    print(model_name, data_name)
    '''
    生成文件夹： result_raw/Cov/exp_mutation_dataset_network
    '''
    base_path = mk_exp_dir(exp_name, data_name, model_name, base_dir)
    exp(model_name, data_name, base_path, seed)
    K.clear_session()


# 将cov 选的不够的补够
def expand_rank_ix(ix, expand_ix_arr, max_select_size):
    ix_arr = ix.tolist()

    for expand_ix in expand_ix_arr:
        if expand_ix not in ix:
            ix_arr.append(expand_ix)
        if len(ix_arr) == max_select_size:
            break

    return np.array(ix_arr, dtype=np.int64)


# 重复数据实验
def exp(model_name, data_name, base_path, seed):
    is_prepare_ps = True
    verbose = 0
    cov_name_list = exp_utils.cov_name_list  # 基于覆盖的方法
    dau_name_arr = exp_utils.dau_name_arr  # 扩增方法，数据增强的方法
    mop_name_arr = exp_utils.mop_name_arr  # 变异方法, 增加了一些冗余的数据
    print(dau_name_arr)
    dau = get_dau(data_name)  # 根据数据集名称得到相应的数据处理类
    mop = Mop()  # 数据 mutation的方法
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)

    model_path = model_conf.get_model_path(data_name, model_name)
    cov_initer = get_cov_initer(x_train, y_train, data_name, model_name)

    ps_path = "{}/ps_data/".format(base_path)
    time_path = "{}/time/".format(base_path)
    os.makedirs(time_path, exist_ok=True)
    os.makedirs(ps_path, exist_ok=True)

    # 选择的测试用例个数的ratio比例

    select_size_ratio_arr = [0.025, 0.05, 0.075, 0.1]
    split_ratio = 0.5  # 将数据集进行了切分
    sample_num = 1
    nb_classes = model_conf.fig_nb_classes(data_name)

    ############################
    # exp
    ############################

    # 构造原始选择集, 以及测试集合
    x_dau_test_arr, y_dau_test_arr, x_val_dict, y_val_dict = get_dau_data(x_test, y_test, dau, dau_name_arr,
                                                                          ratio=split_ratio,
                                                                          shuffle=True)

    select_size_arr = []
    len_data = len(np.concatenate(y_dau_test_arr, axis=0))  # test拼接
    for select_size_ratio in select_size_ratio_arr:
        select_size = int(len_data * select_size_ratio)
        select_size_arr.append(select_size)
    max_select_size = select_size_arr[-1]  # 最大
    print(f"The mop_name_arr is {mop_name_arr}, the max_select_size is {max_select_size}")

    for mop_name in mop_name_arr:  # ["IR", "ORI", "RG", "RP", "CR"]   # 所谓的纯洁的数据集就是原始的数据集，其他的数据集都额外的添加了一些数据
        for i in range(sample_num):  # 每个算子5次,每次的数据组成都不一样
            if mop_name == "ORI":
                x_select = np.concatenate(x_dau_test_arr, axis=0)  # 所有的数据进行了拼接
                y_select = np.concatenate(y_dau_test_arr, axis=0)
            else:
                x_select, y_select = get_mutation_data(mop_name, mop, x_dau_test_arr, y_dau_test_arr, data_name,
                                                       seed=seed, )  # 这里种子是唯一的，保证生成的污染是相同的
            # save_path: 存储的是每个coverage的方法的排序序列
            save_path = os.path.join(ps_path, "{}_" + "{}_{}.npy".format(mop_name, i))
            print("len(x_select)", len(x_select))

            # 优先级排序
            if is_prepare_ps:
                ps_csv_path = os.path.join(time_path, "{}_{}.csv".format(mop_name, i))  # 文件的准备
                df_ps = prepare_cov_ps(cov_name_list, model_path, x_select, save_path, cov_initer, y_select)
                if df_ps is None:
                    ...
                else:
                    if os.path.exists(ps_csv_path):
                        ps_csv_path = os.path.join(time_path, "{}_{}2.csv".format(mop_name, i))
                    df_ps.to_csv(ps_csv_path)
            # 获得优先级序列, 这里需要有_EXT是因为之前生成的文件只是覆盖网络神经单元的数目，
            # 可能要远远小于的要选择的测试用例数目
            for name in cov_name_list:
                fp_extra = save_path.format(name + "_EXT")
                if os.path.exists(fp_extra):
                    continue
                else:
                    fp = save_path.format(name)
                    ix = np.load(fp, allow_pickle=True)
                    ix = np.array(ix[2], dtype=np.int64)
                    if len(ix) < max_select_size:
                        expand_ix_arr = np.random.permutation(len(x_select))
                        ix = expand_rank_ix(ix, expand_ix_arr, max_select_size)  # 每次就存储一个文件，保留最大的选择数目
                    assert len(ix) >= max_select_size
                    np.save(fp_extra, ix)

    for mop_name in mop_name_arr:  # ["IR", "ORI", "RG", "RP", "CR"]   # 所谓的纯洁的数据集就是原始的数据集，其他的数据集都额外的添加了一些数据
        for i in range(sample_num):  # 每个算子5次,每次的数据组成都不一样
            df = None
            csv_path = os.path.join(base_path, "res_{}_{}.csv").format(mop_name, i)  # 存储的原始数据
            if mop_name == "ORI":
                x_select = np.concatenate(x_dau_test_arr, axis=0)  # 所有的数据进行了拼接
                y_select = np.concatenate(y_dau_test_arr, axis=0)
            else:
                x_select, y_select = get_mutation_data(mop_name, mop, x_dau_test_arr, y_dau_test_arr, data_name,
                                                       seed=seed, )  # 这里种子是唯一的，保证生成的污染是相同的
            save_path = os.path.join(ps_path, "{}_" + "{}_{}.npy".format(mop_name, i))
            print("len(x_select)", len(x_select))

            idx_data = {}

            for name in cov_name_list:
                fp_extra = save_path.format(name + "_EXT")

                assert os.path.exists(fp_extra)

                ix = np.load(fp_extra, allow_pickle=True)

                idx_data[name] = ix  # dict: cov_method_name: test case sequence

            # 对于每个选择大小(横轴),计算重训练后的精度(纵轴)
            print("mop_name :{}".format(mop_name), "keys:", idx_data.keys(), "select_size_arr:", select_size_arr)
            for cur_size_ratio, cur_select_size in tqdm(zip(select_size_ratio_arr, select_size_arr)):
                for k, idx in tqdm(idx_data.items()):
                    select_size = cur_select_size
                    name = str(k)
                    # print(type(idx), type(select_size), type(idx[0]))
                    x_s, y_s = x_select[idx][:select_size], y_select[idx][:select_size]  # 截取前Max_size的数组，然后分别统计某些阈值
                    # 去除掉错误的数据
                    x_s = x_s[y_s != -1]
                    y_s = y_s[y_s != -1]
                    assert len(x_s) == len(y_s)
                    effect_len = len(x_s)
                    if len(x_s) == 0:
                        imp_dict = {}
                        retrain_time = 0
                        imp_dict["all"] = 0
                    else:
                        temp_model_path = model_conf.get_temp_model_path(data_name, model_name, name)
                        # return 再训练的精度 ， 再训练的时间
                        imp_dict, retrain_time = retrain_detail(temp_model_path, x_s, y_s, x_train, y_train,
                                                                x_val_dict, y_val_dict,
                                                                model_path, nb_classes,
                                                                verbose=verbose, only_add=False, data_name=data_name,
                                                                seed=seed)
                    cov_trained_csv_data = get_retrain_csv_data(name, imp_dict, retrain_time)
                    sim_rate = 0
                    cov_trained_csv_data["sim_rate"] = sim_rate
                    cov_trained_csv_data["effect_len"] = effect_len
                    cov_trained_csv_data["cur_select_size"] = cur_select_size
                    cov_trained_csv_data["cur_size_ratio"] = cur_size_ratio
                    df = add_df(df, cov_trained_csv_data)
                    df.to_csv(csv_path, index=False)
                    del x_s
            del x_select
            # K.clear_session()


if __name__ == '__main__':
    exp_name = "exp_mutation"  # exp_repeat
    parameters = [("", 0), ("2", 1), ("3", 2)]
    for name, seed in parameters:
        base_dir = "result_raw/Cov{}".format(name)
        for data_name, v_arr in tqdm(model_conf.model_data.items()):
            for model_name in v_arr:
                print(model_name, data_name)
                exec(model_name, data_name, base_dir, seed)
