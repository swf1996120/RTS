import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from utils import model_conf
from statistics_utils.statistics_utils import concat_df, get_order_arr, get_color_arr, get_labels_arr, \
    get_raw_result_dir, \
    get_merge_csv_path, get_base_fig_dir, get_base_table_dir, \
    get_exp_name, get_fig_acc_dir, get_RQ2_table_path
import warnings

warnings.filterwarnings("ignore")


def merge_all_detail(rank_name, cov_name, csv_name, mop_name_arr, sample_num):
    df = None
    for data_name, v_arr in model_conf.model_data.items():
        for model_name in v_arr:
            for rank_type in [rank_name, cov_name]:
                base_path = get_raw_result_dir(rank_type, exp_name, data_name, model_name)
                for mop_name in mop_name_arr:
                    for i in range(sample_num):
                        csv_path = os.path.join(base_path, "res_{}_{}.csv").format(mop_name, i)
                        if not os.path.exists(csv_path):
                            print("warning", "merge_all", csv_path)
                            continue
                            # raise ValueError()
                            # continue
                        df_csv = pd.read_csv(csv_path)
                        print(len(df_csv),base_path)
                        # if "Div" in rank_type:
                        #     assert len(df_csv) == 32  # 8 个排序方法 *4个比例 , 存在DeepDAC的变体 DeepDAC_2
                        # if "Cov" in rank_type:
                        #     assert len(df_csv) == 16  # 4 个排序方法 *4个比例
                        df_s = df_csv[["name", "all", "cur_select_size", "cur_size_ratio", "effect_len"]]
                        df_s = df_s.copy()
                        df_s["mop"] = mop_name
                        df_s["i"] = i
                        df_s["model_name"] = model_name
                        df_s["data_name"] = data_name
                        df_s["pair_name"] = model_conf.get_pair_name(data_name, model_name)
                        df = concat_df(df, df_s)
    merge_csv_path = get_merge_csv_path(csv_name)
    df.to_csv(merge_csv_path, index=False)


# mop_name_arr 所属的mop的属性， sampe_num = 1
def merge_all(mop_name_arr, sample_num):
    # cov, div,  all
    # cov2, div2, all2
    # cov3, div3, all3
    for rank_name, cov_name, csv_name in zip(rank_dir_list, cov_dir_list, merge_csv_name_list):
        merge_all_detail(rank_name, cov_name, csv_name, mop_name_arr, sample_num)


def get_df_merge_all():
    df_merge_all = None
    for merge_csv_name in merge_csv_name_list:  # ["all", "all2", "all3"]
        merge_csv_path = get_merge_csv_path(merge_csv_name)
        df_merge_all_type = pd.read_csv(merge_csv_path)
        df_merge_all = concat_df(df_merge_all, df_merge_all_type)
    return df_merge_all


# def plot_line_figs():
#     fig_acc_dir=get_fig_acc_dir()
#     order_arr = get_order_arr() # all rank method names
#     color_arr = get_color_arr(order_arr) # 图片的
#     df_merge_all = get_df_merge_all()
#     for data_name, v_arr in model_conf.model_data.items():
#         for model_name in v_arr:
#             pair_name = model_conf.get_pair_name(data_name, model_name)
#             df_merge_config = df_merge_all[df_merge_all["pair_name"] == pair_name]
#             df_merge_config_mean = df_merge_config.groupby(["name", "cur_size_ratio"]).agg("mean")
#             df_merge_config_mean = df_merge_config_mean.reset_index()
#             df_merge_config_mean["all_100"] = df_merge_config_mean["all"] * 100
#             # print(df_merge_config_mean)
#             # palette = sns.color_palette(color_arr)
#             # # plt.figure(figsize=(5, 10))
#             # ax = sns.lineplot(x="cur_size_ratio", y="all_100", hue="name", data=df_merge_config_mean, palette=palette,
#             #                   hue_order=order_arr)
#             # handles, labels = ax.get_legend_handles_labels()
#             # ax.legend(handles=handles[:], labels=get_labels_arr(labels[:]))
#             # plt.xticks([0.025, 0.05, 0.075, 0.1], [2.5, 5, 7.5, 10])
#             # # plt.title(pair_name)
#             # plt.xlabel("Percentage of Selected Tests (%)")
#             # plt.ylabel("Accuracy Improvement (%)")
#             # plt.savefig(os.path.join(fig_acc_dir, "{}.png".format(pair_name)), bbox_inches='tight')
#             # plt.savefig(os.path.join(fig_acc_dir, "{}.pdf".format(pair_name)), bbox_inches='tight')
#             # plt.close()
#             # plt.show()
#             # continue


def get_RQ2_tables(res_path):
    for b in [True, False]:
        get_RQ2_tables_detail(res_path, use_p_value=b)


def get_table_frame(ratio, select_str):
    # select_ratio_arr = [2.5, 5, 7.5, 10]
    # select_index_arr = [select_str.format(x) for x in select_ratio_arr]

    data_name_index_arr = []
    model_name_index_arr = []

    for (k, v_arr) in model_conf.model_data.items():
        for v in v_arr:
            data_name_index_arr.append(k)
            model_name_index_arr.append(v)
    select_index_arr = [select_str.format(ratio)] * len(data_name_index_arr)

    index_arr = [select_index_arr, data_name_index_arr, model_name_index_arr]

    base_rank_columns_arr = get_order_arr()
    rank_columns_arr = []
    rank_columns_arr2 = []
    for rank in base_rank_columns_arr:
        rank_columns_arr2.append("acc")
        rank_columns_arr.append(rank)
        if rank != "DeepDAC":
            rank_columns_arr2.append("p_value")
            rank_columns_arr.append(rank)

    columns = [rank_columns_arr, rank_columns_arr2]
    # print(columns)
    df = pd.DataFrame(
        index=index_arr,
        columns=columns)
    return df


def get_RQ2_tables_detail(res_path, use_p_value=True):
    if not use_p_value:
        res_path = res_path + "bool.csv"
    select_str = "Select {}%"
    df_all = None
    select_ratio_arr = [2.5, 5, 7.5, 10]
    select_size_ratio_arr = [0.025, 0.05, 0.075, 0.1]
    rank_name_list = get_order_arr()
    rank_name_list.remove("DeepDAC")
    df_merge = get_df_merge_all()  # df_merge: result/all,  result/all2.csv, 所有结果合并
    for ratio_p, cur_size_ratio in zip(select_ratio_arr, select_size_ratio_arr):
        df_ratio = get_table_frame(ratio_p, select_str)
        df_merge_ratio = df_merge[df_merge["cur_size_ratio"] == cur_size_ratio]
        for data_name, v_arr in model_conf.model_data.items():
            for model_name in v_arr:
                pair_name = model_conf.get_pair_name(data_name, model_name)
                df_merge_ratio_config = df_merge_ratio[df_merge_ratio["pair_name"] == pair_name]
                div_acc_arr = df_merge_ratio_config[df_merge_ratio_config["name"] == "DeepDAC"]["all"]
                div_acc_arr = div_acc_arr.values.tolist()
                mean_acc = np.mean(div_acc_arr)
                df_index = (select_str.format(ratio_p), data_name, model_name)
                df_ratio.loc[df_index, ("DeepDAC", "acc")] = mean_acc
                for rank_name in rank_name_list:
                    rank_acc_arr = df_merge_ratio_config[df_merge_ratio_config["name"] == rank_name]["all"]
                    # print(len(rank_acc_arr))
                    rank_acc_arr = rank_acc_arr.values.tolist()
                    # print(len(rank_acc_arr), len(div_acc_arr), rank_name, ratio_p, model_name, data_name)
                    res = stats.wilcoxon(rank_acc_arr, div_acc_arr, alternative="less")
                    # print(len(rank_acc_arr), len(div_acc_arr))
                    mean_acc = np.mean(rank_acc_arr)
                    df_ratio.loc[df_index, (rank_name, "acc")] = mean_acc
                    if use_p_value:
                        df_ratio.loc[df_index, (rank_name, "p_value")] = res[1]
                    else:
                        if res[1] < 0.05:
                            comp_res = "好"
                        elif res[1] >= 0.95:
                            comp_res = "差"
                        else:
                            comp_res = ""
                        df_ratio.loc[df_index, (rank_name, "p_value")] = comp_res
                    # if res[1] >= 0.05:
                    #     print(data_name, model_name, rank_name, res[1], ratio_p)
                    # print(rank_acc_arr.values, div_acc_arr.values)
        df_all = concat_df(df_all, df_ratio)
    # print(df_all)
    df_all.to_csv(res_path)


# config
exp_name = get_exp_name()
base_fig_dir = get_base_fig_dir()
base_res_dir = get_base_table_dir()
os.makedirs(base_fig_dir, exist_ok=True)
os.makedirs(base_res_dir, exist_ok=True)
rank_dir_list = ["Div", "Div2", "Div3",]
cov_dir_list = ["Cov", "Cov2", "Cov3",]
merge_csv_name_list = ["all", "all2", "all3"]


def RQ2():
    # Step1: merge all rank results
    mop_name_arr = ["ORI", "RG", "RP", "IR", "CR"]
    merge_all(mop_name_arr,sample_num=1)

    # tab
    RQ2_table_path = get_RQ2_table_path()
    get_RQ2_tables(RQ2_table_path)
