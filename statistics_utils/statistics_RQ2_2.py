import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import model_conf
from statistics_utils.statistics_RQ1 import get_max_size
from statistics_utils.statistics_utils import concat_df, get_order_arr, get_raw_result_dir, \
    get_merge_csv_path, get_base_fig_dir, get_base_table_dir, get_exp_name, get_RQ3_line_fig_table_dir, get_color_dict, get_RQ3_line_fig_dir
import warnings

warnings.filterwarnings("ignore")


def get_df_merge_all():
    df_merge_all = None
    for merge_csv_name in merge_csv_name_list:
        merge_csv_path = get_merge_csv_path(merge_csv_name)
        df_merge_all_type = pd.read_csv(merge_csv_path)
        df_merge_all = concat_df(df_merge_all, df_merge_all_type)
    return df_merge_all


def get_table_frame(ratio, select_str, mop_arr):
    # select_ratio_arr = [2.5, 5, 7.5, 10]
    # select_index_arr = [select_str.format(x) for x in select_ratio_arr]

    data_name_index_arr = []
    model_name_index_arr = []

    for (k, v_arr) in model_conf.model_data.items():
        for v in v_arr:
            for i in range(len(mop_arr)):
                data_name_index_arr.append(k)
                model_name_index_arr.append(v)
    select_index_arr = [select_str.format(ratio)] * len(data_name_index_arr)
    mop_index_arr = mop_arr * 8
    index_arr = [select_index_arr, data_name_index_arr, model_name_index_arr, mop_index_arr]
    base_rank_columns_arr = get_order_arr()
    # rank_columns_arr = []
    # rank_columns_arr2 = []
    # for rank in base_rank_columns_arr:
    #     rank_columns_arr2.append("acc")
    #     rank_columns_arr.append(rank)
    #     if rank != "DeepDiv":
    #         rank_columns_arr2.append("p_value")
    #         rank_columns_arr.append(rank)

    columns = base_rank_columns_arr
    # print(columns)
    df = pd.DataFrame(
        index=index_arr,
        columns=[columns])
    return df


def get_rank_order(rank_value_arr):
    value_sort = sorted(list(set(rank_value_arr)), reverse=True)
    rank_order = []
    for rank_acc in rank_value_arr:
        o = value_sort.index(rank_acc)
        rank_order.append(o + 1)
    return rank_order


def get_error_num(max_size, error_dir, rank_name, mop_name, i=0):
    error_path = os.path.join(error_dir, "{}_".format(rank_name) + "{}" + "_{}.npy".format(i))

    count = 0
    ix_arr = np.load(error_path.format(mop_name))
    for i in range(len(ix_arr)):
        if ix_arr[i] >= max_size:
            continue
        count += 1
    return count


def get_acc_order_table(res_dir, rank_name_list, select_str):
    res_path = os.path.join(res_dir, "order_acc.csv")
    df_all = None
    df_merge = get_df_merge_all()
    for ratio_p, cur_size_ratio in tqdm(zip(select_ratio_arr, select_size_ratio_arr)):
        df_ratio = get_table_frame(ratio_p, select_str, mop_name_arr)
        df_merge_ratio = df_merge[df_merge["cur_size_ratio"] == cur_size_ratio]
        for data_name, v_arr in model_conf.model_data.items():
            for model_name in v_arr:
                pair_name = model_conf.get_pair_name(data_name, model_name)
                df_merge_ratio_config = df_merge_ratio[df_merge_ratio["pair_name"] == pair_name]
                for mop_name in mop_name_arr:
                    df_merge_ratio_config_mop = df_merge_ratio_config[df_merge_ratio_config["mop"] == mop_name]
                    rank_value_arr = []
                    for rank_name in rank_name_list:
                        rank_acc = df_merge_ratio_config_mop[df_merge_ratio_config_mop["name"] == rank_name]["all"]
                        rank_acc = rank_acc.values.tolist()
                        rank_acc = np.mean(rank_acc)
                        rank_value_arr.append(rank_acc)
                        # print(rank_name, rank_acc)
                    rank_order = get_rank_order(rank_value_arr)
                    for name, o in zip(rank_name_list, rank_order):
                        df_index = (select_str.format(ratio_p), data_name, model_name, mop_name)
                        df_ratio.loc[df_index, name] = o
        df_all = concat_df(df_all, df_ratio)
    df_all.to_csv(res_path)


def get_error_num_order_table(res_dir, rank_name_list, select_str):
    arr1 = ["DeepDiv", "Gini", "CES", "LSA", "MAX_P", "Random"]
    arr2 = ["NAC", "NBC", "SNAC", "TKNC"]
    sp_num = 0
    df_all = None
    res_path = os.path.join(res_dir, "order_fault.csv")
    for ratio_p, cur_size_ratio in tqdm(zip(select_ratio_arr, select_size_ratio_arr)):
        df_ratio = get_table_frame(ratio_p, select_str, mop_name_arr)
        for data_name, v_arr in model_conf.model_data.items():
            max_size = get_max_size(data_name, cur_size_ratio)
            for model_name in v_arr:
                for mop_name in mop_name_arr:
                    rank_value_arr = []
                    for rank_name in rank_name_list:
                        if rank_name in arr1:
                            base_path = get_raw_result_dir(rank_dir, exp_name, data_name, model_name)
                        elif rank_name in arr2:
                            base_path = get_raw_result_dir(cov_dir, exp_name, data_name, model_name)
                        else:
                            raise ValueError()
                        # 每个位置有几个不一样的错误    0,1,3,1,4,2,0,3
                        error_dir = "{}/fault/".format(base_path)
                        error_num = get_error_num(max_size, error_dir, rank_name, mop_name, i=sp_num)
                        rank_value_arr.append(error_num)
                    rank_order = get_rank_order(rank_value_arr)
                    for name, o in zip(rank_name_list, rank_order):
                        df_index = (select_str.format(ratio_p), data_name, model_name, mop_name)
                        # print(df_index, name)
                        df_ratio.loc[df_index, name] = o
        df_all = concat_df(df_all, df_ratio)
    df_all.to_csv(res_path)


def get_diverse_error_order_table(res_dir, rank_name_list, select_str):
    arr1 = ["DeepDiv", "Gini", "CES", "LSA", "MAX_P", "Random"]
    arr2 = ["NAC", "NBC", "SNAC", "TKNC"]
    sp_num = 0
    df_all = None
    res_path = os.path.join(res_dir, "order_diverse.csv")
    for ratio_p, cur_size_ratio in tqdm(zip(select_ratio_arr, select_size_ratio_arr)):
        df_ratio = get_table_frame(ratio_p, select_str, mop_name_arr)
        for data_name, v_arr in model_conf.model_data.items():
            max_size = get_max_size(data_name, cur_size_ratio)
            for model_name in v_arr:
                for mop_name in mop_name_arr:
                    rank_value_arr = []
                    for rank_name in rank_name_list:
                        if rank_name in arr1:
                            base_path = get_raw_result_dir(rank_dir, exp_name, data_name, model_name)
                        elif rank_name in arr2:
                            base_path = get_raw_result_dir(cov_dir, exp_name, data_name, model_name)
                        else:
                            raise ValueError()
                        error_dir = "{}/diverse_error/".format(base_path)
                        # 每个位置有几个不一样的错误    0,1,3,1,4,2,0,3
                        error_num = get_error_num(max_size, error_dir, rank_name, mop_name, i=sp_num)
                        rank_value_arr.append(error_num)
                    rank_order = get_rank_order(rank_value_arr)
                    for name, o in zip(rank_name_list, rank_order):
                        df_index = (select_str.format(ratio_p), data_name, model_name, mop_name)
                        # print(df_index, name)
                        df_ratio.loc[df_index, name] = o
        df_all = concat_df(df_all, df_ratio)
    df_all.to_csv(res_path)


def get_tables(res_dir):
    rank_name_list = get_order_arr()
    # 获取排序顺序
    get_acc_order_table(res_dir, rank_name_list, select_str)
    get_error_num_order_table(res_dir, rank_name_list, select_str)
    get_diverse_error_order_table(res_dir, rank_name_list, select_str)


def merge_tables(res_dir):
    res_path_acc = os.path.join(res_dir, "order_acc.csv")
    res_path_fault = os.path.join(res_dir, "order_fault.csv")
    res_path_diverse = os.path.join(res_dir, "order_diverse.csv")
    res_path_all = os.path.join(res_dir, "order_all.csv")

    res_dict = {
        "acc": res_path_acc,
        "fault": res_path_fault,
        "diverse": res_path_diverse
    }
    df_order_merge = None
    for k, res_path in res_dict.items():
        df_res = pd.read_csv(res_path, index_col=[0])
        # df_res = df_res.reset_index(col_fill=["ratio", "data_name", "model_name", "mop_name"])
        # df_res.index.names = ["ratio", "data_name", "model_name", "mop_name"]
        print(df_res)
        df_res_group = df_res.groupby(level=0).mean()
        df_res_group["type"] = k
        print(df_res_group)
        df_order_merge = concat_df(df_order_merge, df_res_group)
    df_order_merge.to_csv(res_path_all)


def plot_figs(res_dir, fig_dir):
    res_path_all = os.path.join(res_dir, "order_all.csv")
    df_all = pd.read_csv(res_path_all, index_col=[0])
    df_all = df_all.reset_index()
    color_dict = get_color_dict()
    for ratio_p, cur_size_ratio in tqdm(zip(select_ratio_arr, select_size_ratio_arr)):
        index_name = select_str.format(ratio_p)
        df_index = df_all[df_all["index"] == index_name]
        rank_name_list = get_order_arr()
        for rank_name in rank_name_list:
            plt.plot(df_index["type"], df_index[rank_name], label=rank_name, color=color_dict[rank_name])
        plt.yticks(range(1, 11))
        ax = plt.gca()

        ax.invert_yaxis()
        plt.ylabel("order")
        plt.xlabel("type")
        plt.legend()
        plt.title(index_name)
        # plt.show()
        plt.savefig(os.path.join(fig_dir, "{}.png".format(ratio_p)))
        plt.close()


# config
exp_name = get_exp_name()
base_fig_dir = get_base_fig_dir()
base_res_dir = get_base_table_dir()
os.makedirs(base_fig_dir, exist_ok=True)
os.makedirs(base_res_dir, exist_ok=True)
rank_dir_list = ["Div", "Div2"]
cov_dir_list = ["Cov", "Cov2"]
merge_csv_name_list = ["all", "all2"]
mop_name_arr = ["ORI", "RG", "RP", "IR", "CR"]
rank_dir = "Div"
cov_dir = "Cov"
select_size_ratio_arr = [0.025, 0.05, 0.075, 0.1]
select_ratio_arr = [2.5, 5, 7.5, 10]
select_str = "Select {}%"


def RQ3_2():
    # tab
    RQ3_table_dir = get_RQ3_line_fig_table_dir()
    os.makedirs(RQ3_table_dir, exist_ok=True)
    # get_tables(RQ3_table_dir)
    # merge_tables(RQ3_table_dir)
    fig_dir = get_RQ3_line_fig_dir()
    os.makedirs(fig_dir, exist_ok=True)
    plot_figs(RQ3_table_dir, fig_dir)
