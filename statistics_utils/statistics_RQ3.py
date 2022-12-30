import os
import pandas as pd
from tqdm import tqdm

from utils import model_conf
import numpy as np

from statistics_utils.statistics_utils import get_exp_name, get_base_fig_dir, get_base_table_dir, get_order_arr, \
    get_RQ1_table_path, get_raw_result_dir, num_to_str, get_RQ3_table_path


def get_max_size(data_name, ratio):
    ratio_type_dict = {
        0.025: 0,
        0.05: 1,
        0.075: 2,
        0.1: 3
    }
    ratio_type = ratio_type_dict[ratio]
    if data_name == model_conf.svhn:
        arr = [2603, 5206, 7809, 10412]  # 这个地方可能要改
        # return 10412
        return arr[ratio_type]
    else:
        # return
        arr = [1000, 2000, 3000, 4000]
        return arr[ratio_type]


def get_table_frame(index_arr):
    columns = []
    model_data = model_conf.model_data
    for (k, v_arr) in model_data.items():
        for v in v_arr:
            columns.append(model_conf.get_pair_name(k, v))
    df = pd.DataFrame(
        index=index_arr,
        columns=[columns])
    return df


def get_error_num(max_size, error_dir, rank_name, i=0):
    error_path = os.path.join(error_dir, "{}_".format(rank_name) + "{}" + "_{}.npy".format(i))

    count = 0  # 统计的是在每种情况下
    for name in mop_name_arr:
        ix_arr = np.load(error_path.format(name))
        for i in range(len(ix_arr)):
            if ix_arr[i] >= max_size:
                continue
            count += 1
    return count


def statistics(csv_path, ratio):
    arr1 = ["DeepDAC", "DeepDAC-var1", "DeepDAC-var2", "DeepDAC-var3"]
    sp_num = 0
    rank_name_list = ["DeepDAC", "DeepDAC-var1", "DeepDAC-var2", "DeepDAC-var3"]
    df = get_table_frame(rank_name_list)
    for data_name, v_arr in tqdm(model_conf.model_data.items()):
        max_size = get_max_size(data_name, ratio)
        for model_name in v_arr:
            print(model_name, data_name)
            pair_name = model_conf.get_pair_name(data_name, model_name)
            for rank_name in rank_name_list:
                if rank_name in arr1:
                    base_path = get_raw_result_dir(rank_dir, exp_name, data_name, model_name)
                else:
                    raise ValueError()
                # base_path 是定位到具体的被测数据集 模型 文件夹
                # 每个位置有几个不一样的错误    0,1,3,1,4,2,0,3
                error_dir = "{}/fault/".format(base_path)
                # error_num 是所有模型情况下的失效测试用例总数
                error_num = get_error_num(max_size, error_dir, rank_name, i=sp_num)
                # 保留三位小数
                df.loc[[rank_name], [pair_name]] = num_to_str(error_num * 100 / (max_size * len(mop_name_arr)), 3)
    df.to_csv(csv_path)


exp_name = get_exp_name()
base_fig_dir = get_base_fig_dir()
base_res_dir = get_base_table_dir()
os.makedirs(base_fig_dir, exist_ok=True)
os.makedirs(base_res_dir, exist_ok=True)
rank_dir = "Div"
mop_name_arr = ["ORI", "RG", "RP", "IR", "CR"]


def RQ3():
    for ratio in [0.025, 0.05, 0.075, 0.1]:
        csv_path = get_RQ3_table_path() + "_{}.csv".format(ratio)
        statistics(csv_path, ratio)
