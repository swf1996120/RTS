import os

import pandas as pd
import numpy as np


## 获取路径
def get_base_fig_dir():
    base_fig_dir = "result/fig"
    return base_fig_dir


def get_RQ2_figs_dir():
    return os.path.join(get_base_fig_dir(), "diverse_errors")


def get_RQ3_figs_dir():
    return os.path.join(get_base_fig_dir(), "acc")


def get_RQ3_line_fig_dir():
    return os.path.join(get_base_fig_dir(), "order")


def get_base_table_dir():
    base_res_dir = "result/table"
    return base_res_dir


def get_RQ1_table_path():
    return os.path.join(get_base_table_dir(), "RQ1.csv")


def get_RQ1_1_table_path():
    return os.path.join(get_base_table_dir(), "RQ1_1.csv")


def get_RQ2_table_path():
    return os.path.join(get_base_table_dir(), "RQ2.csv")


def get_RQ3_table_path():
    return os.path.join(get_base_table_dir(), "RQ3")


def get_RQ3_line_fig_table_dir():
    return os.path.join(get_base_table_dir(), "order")


def get_exp_name():
    exp_name = "exp_mutation"
    return exp_name


def get_raw_result_dir(dir_name, exp_name, data_name, model_name):
    base_path = "result_raw/{}/{}_{}_{}".format(dir_name, exp_name, data_name, model_name)
    return base_path


def get_merge_csv_path(csv_name):
    return "result/{}.csv".format(csv_name)


def get_fig_acc_dir():
    return "result/fig"


## 画图画表
def get_color_dict():
    color_dict = {
        "LSA": "C0",
        "DeepDiv": "C9",
        "CES": "C1",
        "Random": "black",
        "Gini": "C2",
        "MAX_P": "C4",
        "Theo": "navy",
        "NAC": "C5",
        "NBC": "C6",
        "SNAC": "C7",
        "TKNC": "C8",
        "DeepDAC": "C10",
        "DeepDAC_2": "crimson",
        "Test": "crimson",
    }
    return color_dict


def num_to_str(num, trunc=2):
    return format(num, '.{}f'.format(trunc))


def get_color_arr(base_order_arr):
    color_dict = get_color_dict()
    color_arr = []
    for name in base_order_arr:
        color_arr.append(color_dict[name])
    return color_arr


def get_label(name):
    if name == "MAX_P":
        return "MAXp"
    elif name == "DeepDiv":
        return "ATS"
    elif name == "Random":
        return "RS"
    elif name == "DeepDAC":
        return "RTS"
    else:
        return name


def get_labels_arr(label_arr):
    arr = []
    for lb_name in label_arr:
        lb_name = get_label(lb_name)
        arr.append(lb_name)
    return arr


def get_order_arr():
    base_rank_arr = ["DeepDAC", "DeepDiv", "Gini", "CES", "LSA", "MAX_P", "Kmeas_sample",
                     "KCenterGreedy", "var_rank", "ood_gini", "msp_gini", "entrory_dropout_rank", "BALDDropout",
                     "BADGE_sample", "CLUE_sample"]  # DeepDAC_2 ,DeepDAC
    base_cov_arr = ["NAC", "NBC", "SNAC", "TKNC", ]
    base_random_arr = ["Random"]
    base_order_arr = base_rank_arr + base_cov_arr + base_random_arr
    return base_order_arr


###
# 工具函数
def add_df(df, csv_data):
    if df is None:  # 如果是空的
        df = pd.DataFrame(csv_data, index=[0])
    else:
        df.loc[df.shape[0]] = csv_data
    return df


def concat_df(df1, df2):
    if df1 is None:
        return df2
    else:
        df = pd.concat([df1, df2], axis=0)
    return df
