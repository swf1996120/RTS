import bisect
import os
import time
from collections import defaultdict
import numpy as np
from ATS.selection.AbsSelectStrategy import AbsSelectStrategy
import pandas as pd
from utils.utils import get_data_by_label_with_idx


# 根据每个数据的错误模式覆盖大小排序
# 在区域内的选取的数据有顺序
# 区域外的数据无顺序

# 使用排序表优化了cam
class PrioritySelectStrategy(AbsSelectStrategy):

    def sort_ix_by_len(self, x_all, x_all_len):
        assert len(x_all) == len(x_all_len)
        dic = {}
        for ix, ix_len in zip(x_all, x_all_len):
            dic[ix] = ix_len
        assert len(dic) == len(x_all)  # 确保没有重复的下标

        sort_list_dict = sorted(dic.items(), key=lambda x: x[1], reverse=True)  # 由大到小排序
        sort_ix, sort_len = zip(*sort_list_dict)
        return sort_ix, sort_len

    def get_priority_sequence(self, Tx, Ty, n, M, base_path=None,  prefix=None, is_save_ps=False,th=0.001):
        Xr_select, Xr_select_len, Xr_others, Xr_others_len, c_arr, max_size_arr, idx_others = \
            self.get_max_coverage_sequence(Tx,
                                           Ty,
                                           n,
                                           M,
                                           base_path=base_path,
                                           use_add=True,
                                           is_save_ps=is_save_ps,
                                           prefix=prefix,
                                           th=th, )

        x_all = np.concatenate(Xr_select, axis=0)
        x_all_len = np.concatenate(Xr_select_len, axis=0)
        sort_select_ix, sort_select_len = self.sort_ix_by_len(x_all, x_all_len)

        x_others = np.concatenate(Xr_others, axis=0)
        x_others_len = np.concatenate(Xr_others_len, axis=0)
        sort_others_ix, sort_others_len = self.sort_ix_by_len(x_others, x_others_len)

        np.random.seed(0)
        idx_others = np.concatenate(idx_others, axis=0)
        shuffle_ix = np.random.permutation(len(idx_others))
        idx_others = idx_others[shuffle_ix]

        sort_ix = np.concatenate([sort_select_ix, sort_others_ix, idx_others], axis=0)  # 选择的,选择剩下的,被过滤掉的
        # assert len(sort_ix) == len(Ty)
        # assert len(sort_ix) == len(set(sort_ix))
        return sort_ix, c_arr, max_size_arr

    # 没有target_size
    # 目的是选取最大覆盖,而非均匀选取
    # 获取获取优先级序列,
    def get_max_coverage_sequence(self, Tx, Ty, n, M, base_path=None, use_add=False, is_save_ps=False, prefix="",
                                  th=0.001):
        df = None
        csv_data = {}
        c_arr = []  # 存储覆盖率
        max_size_arr = []  # 存储最大覆盖个数
        Xr_select = []  # 存储选择的下标
        Xr_select_len = []  # 存储选择的下标对应覆盖长度

        Xr_others = []  # 存储选择后剩下的下标
        Xr_others_len = []  # 存储选择后剩下的下标的覆盖长度

        idx_others = []
        for i in range(n):
            csv_data["label"] = i
            # 返回值,返回一个选择标记和选择数据编号集
            Tx_i, Ty_i, T_idx_arr = get_data_by_label_with_idx(Tx, Ty, i)  # 返回所有数据的绝对下标
            if len(Tx_i) == 0:
                continue  # 跳过当前迭代，继续下一个迭代
            Tx_prob_matrixc = M.predict(Tx_i)  # 原始预测向量矩阵
            # 返回T_idx_arr的相对下标
            S_up, rel_idx_up, S_mid, rel_idx_mid, S_low, rel_idx_low \
                = self.cluster_test_step.split_data_region_with_idx(Tx_prob_matrixc, i,
                                                                    np.array(range(len(Tx_prob_matrixc))))

            rel_select_idx, idx_max_diff_arr, rel_others_idx, ctm_max_diff_arr, C_select_i_map, max_cov_point_size = \
                self.get_priority_sequence_detail(
                    S_mid, n, i,
                    rel_idx_mid,
                    use_add=use_add,
                    th=th, )
            abs_ix_up = T_idx_arr[rel_idx_up]
            idx_others.append(abs_ix_up)
            # 数据编号
            abs_idx = T_idx_arr[rel_select_idx]
            abs_idx_others = T_idx_arr[rel_others_idx]
            # assert len(set(abs_idx) & set(abs_idx_others)) == 0
            # assert len(set(abs_ix_up) & set(abs_idx_others)) == 0
            # assert len(set(abs_idx) & set(abs_ix_up)) == 0
            Xr_select.append(abs_idx)  # 所选数据编号
            Xr_select_len.append(idx_max_diff_arr)
            Xr_others.append(abs_idx_others)
            Xr_others_len.append(ctm_max_diff_arr)
            csv_data["len(S_up)"] = len(S_up)
            csv_data["len(S_mid)"] = len(S_mid)
            # 覆盖率
            s_pq_arr = self.pattern_fitness_step.get_cov_length_map(C_select_i_map, n, i, )
            # 计算第i个lable总覆盖长度与覆盖率率
            s, c_i = self.pattern_fitness_step.get_cov_s_and_c(s_pq_arr, n)
            print("cov rate ", c_i)
            csv_data["div"] = c_i
            c_arr.append(c_i)  # 该lable的覆盖率

            # 最大覆盖个数:
            max_size_arr.append(max_cov_point_size)
            csv_data["max_cov"] = max_cov_point_size
            if df is None:  # 如果是空的
                df = pd.DataFrame(csv_data, index=[0])
            else:
                df.loc[df.shape[0]] = csv_data
            if base_path is not None:
                csv_path = base_path + "/data_select.csv"
                df.to_csv(csv_path, index=False)
        if is_save_ps:
            ps_path = base_path + "/ps"
            ps_path_all = base_path + "/ps_all"
            os.makedirs(ps_path, exist_ok=True)
            os.makedirs(ps_path_all, exist_ok=True)
            for i in range(n):
                idx_arr = Xr_select[i]
                if prefix == "":
                    save_path = ps_path + "/{}.npy".format(i)
                else:
                    save_dir = ps_path + "/{}".format(prefix)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = save_dir + "/{}.npy".format(i)
                np.save(save_path, idx_arr)
            x_all_idx = np.concatenate(Xr_select, axis=0)
            save_path2 = ps_path_all + "/{}all.npy".format(prefix)
            np.save(save_path2, x_all_idx)
        return Xr_select, Xr_select_len, Xr_others, Xr_others_len, c_arr, max_size_arr, idx_others

    # 排序表版本
    def get_priority_sequence_detail(self, S_mid, n, i, idx_mid, use_add=False, th=0.001, use_fine=False):
        print("th", th, "use_add", use_add)
        temp_c_arr = []
        rel_mid_idx = []  # 使用idx_mid的相对编号
        idx_max_diff_arr = []

        ctm_mid_idx = []  # 当小于阈值时,剩下的按照ctm排序
        ctm_max_diff_arr = []

        C_select_i_map = defaultdict(list)  # 如果添加初始的数据,一定要把这里改了
        X_no_i = []
        # 按i p q 获取覆盖对 ipq a b
        ck_list_map = self.ats_utils.get_ck_list_map(S_mid, n, i)
        ### 将 get_ck_list_map reshpe一下
        C_Xr_list = list(map(list, zip(*ck_list_map.values())))  # 行smaples 列pq_nums 每行为该用例x对应的在所有pq维度下的覆盖对
        # if len(idx_mid) != len(C_Xr_list):
        #     raise ValueError("len idx not eq len data")  # "下标个数与数据个数不符"
        ### step1 记录Ctm表格
        # 1.记录每个用例的下标及其最大覆盖长度
        all_rank_arr = []
        for ii, xi in enumerate(C_Xr_list):
            l = np.array(xi)[:, 4].sum()
            all_rank_arr.append([ii, l])
        # 2. 对覆盖长度排序
        all_rank_arr.sort(key=lambda x: (x[1]), reverse=False)  # 由小到大排序
        # 3. 获得排序后的下标及覆盖长度
        all_rank_idx = np.array(all_rank_arr)[:, 0].astype("int").tolist()
        all_rank_cov_len = np.array(all_rank_arr)[:, 1].tolist()
        # 点选择
        print(i, "************************************", time.time())
        max_size = 0
        while True:
            max_s_i = 0  # 最大覆盖长度
            max_idx = 0  # 最大覆盖长度元素的下标
            max_s_diff = 0
            max_Cr_select_i_map = None  # 最大覆盖集合C
            # step 1 计算选择集的覆盖长度
            s_c_select = self.pattern_fitness_step.get_cov_pair_map_len(C_select_i_map, n, i)  # 当前s_select的覆盖长度
            # 计算覆盖率
            c = self.pattern_fitness_step.get_cov_c(s_c_select, n)
            temp_c_arr.append(c)
            # print("# 当前覆盖率: {}".format(c), time.time())
            # print("# 当前覆盖长: {}".format(s_c_select), time.time())
            # print("# current no select data point", len(X_no_i))
            # 要插入排序的用例
            all_rank_cov_len_copy = all_rank_cov_len.copy()
            all_rank_idx_copy = all_rank_idx.copy()
            for iix in range(len(all_rank_idx) - 1, -1, -1):  # 由大到小遍历
                j = all_rank_idx[iix]  # 当前所选数据编号  iix是给排序后的数组用的, j是给原数组用的, j是用例编号,iix是排序后顺序
                if j in X_no_i:
                    continue
                Cx = C_Xr_list[j]  # 当前数据
                Cx_insert = self.pattern_fitness_step.union_cov_maps(Cx, C_select_i_map, X_no_i)
                # step 3 计算并集
                # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^2", time.time())
                Cx_union = self.pattern_fitness_step.statistic_union_map(Cx_insert, n, i)
                # print(Cx_union, "===")
                # step 3 计算并集的覆盖长度
                # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^3", time.time())
                s_c_union = self.pattern_fitness_step.get_cov_pair_map_len(Cx_union, n, i)
                # print(s_c_union, "===")
                # step 4 剔除加入后没有变化的数据x
                s_diff = s_c_union - s_c_select  # 覆盖长度差值
                if abs(s_diff) <= th:  # 浮点数计算,如果没有大于一定阈值,认为是没变化
                    X_no_i.append(j)
                elif s_c_union > s_c_select:  # 正常流程,即合并后大于上一次的值
                    # 如果大于最大值,则更新最大值
                    if s_c_union > max_s_i:
                        max_s_i = s_c_union  # 更新最大覆盖长度
                        max_idx = j  # 更新所选数据
                        max_Cr_select_i_map = Cx_union  # 更新C_select
                        max_s_diff = s_diff
                    # step5 提前终止算法
                    # 判断此前的最大值,是否大于之前的最大值
                    if iix != 0 and max_s_diff >= all_rank_cov_len[iix - 1]:
                        # 如果这次增加的了,比上次增加的还多(上次增加的只能<=这次)
                        # print("selected lables: ", len(rel_mid_idx), "early stop in: ", iix, "cur max_s_diff", max_s_diff)
                        break
                    else:
                        # 否则,就要更新排序表
                        all_rank_cov_len_copy.remove(all_rank_cov_len[iix])  # 移除对应的编号 和长度
                        all_rank_idx_copy.remove(j)
                        ins_idx = bisect.bisect(all_rank_cov_len_copy, s_diff)
                        all_rank_idx_copy.insert(ins_idx, j)
                        all_rank_cov_len_copy.insert(ins_idx, s_diff)
                        # print(iix, "--->", ins_idx)
                else:
                    X_no_i.append(j)
            if max_s_i != 0:  # 如果本次循环找到了一个新的可以增加覆盖的点
                rel_mid_idx.append(max_idx)  # 添加该数据编号
                idx_max_diff_arr.append(max_s_diff)
                # print(len(rel_mid_idx), max_s_diff)
                X_no_i.append(max_idx)  # 该数据被选过
                C_select_i_map = max_Cr_select_i_map.copy()  # 更新添加后的覆盖对集合
                all_rank_idx = all_rank_idx_copy.copy()
                all_rank_cov_len = all_rank_cov_len_copy.copy()
                if max_s_diff < 0.005:
                    pass
            else:  # 没找到一个新的点,则循环结束
                max_size = len(rel_mid_idx)
                if use_add:
                    s = time.time()
                    # 直接用最开始的ctm
                    # if not use_fine:
                    #     # 记录最大选择点的数量
                    #     Xr_ctm_idx = list(set(X_no_i) - set(rel_mid_idx))  # 没选过,但又被淘汰了的数据编号
                    #     C_Xr_CTM_list = np.array(C_Xr_list)[Xr_ctm_idx]  # 剩下的这些数据里,选覆盖大的
                    #     # 排序 C_Xr_CTM_list, 把每行第4个数加起来后排序
                    #     sorted_arr = []
                    #     for i_pq, x_pq in zip(Xr_ctm_idx, C_Xr_CTM_list):
                    #         len_s = np.array(x_pq)[:, 4].sum()
                    #         sorted_arr.append([i_pq, len_s])
                    #     sorted_arr.sort(key=lambda x: (x[1]), reverse=True)  # 由大到小排序
                    #     sorted_arr = np.array(sorted_arr)
                    #     select_ctm_x = sorted_arr[:, 0].astype("int").tolist()
                    #     rel_mid_idx += select_ctm_x  # 合并至原始下标内
                    #     idx_max_diff_arr += [0] * len(select_ctm_x)
                    # else:

                    # 对所有剩下的数据再算一遍
                    # Xr_ctm_idx = list(set(X_no_i) - set(rel_mid_idx))
                    # ctm_len_arr = []
                    # for ctm_ix in Xr_ctm_idx:
                    #     Cx = C_Xr_list[ctm_ix]  # 当前数据
                    #     Cx_insert = union_cov_maps(Cx, C_select_i_map, X_no_i)
                    #     Cx_union = statistic_union_map(Cx_insert, n, i)
                    #     s_c_union = get_cov_pair_map_len(Cx_union, n, i)
                    #     s_diff = s_c_union - s_c_select  # 覆盖长度差值
                    #     ctm_len_arr.append(s_diff)
                    # ctm_mid_idx = Xr_ctm_idx
                    # ctm_max_diff_arr = ctm_len_arr

                    Xr_ctm_idx = list(set(X_no_i) - set(rel_mid_idx))
                    # 剩下的现在是多长就多长
                    for iix in all_rank_idx:
                        j = all_rank_idx[iix]
                        if j in Xr_ctm_idx:
                            ctm_mid_idx.append(j)
                            cov_len = all_rank_cov_len[j]
                            ctm_max_diff_arr.append(cov_len)
                    e = time.time()
                    print("ctm time: ", e - s)
                    break
                else:
                    break

        assert len(rel_mid_idx) == len(idx_max_diff_arr)
        idx_mid = np.array(idx_mid)
        Xr_select_i = idx_mid[rel_mid_idx]  # 根据相对编号获得原始数据的编号
        Xr_others_i = idx_mid[ctm_mid_idx]  # 根据相对编号获得原始数据的编号
        # if len(Xr_select_i) != len(set(Xr_select_i)):
        #     raise ValueError("some data points  repeatly select")  # "有数据点被重复选取了"
        # if len(Xr_others_i) != len(set(Xr_others_i)):
        #     raise ValueError("some data points  repeatly select")  # "有数据点被重复选取了"
        # assert len(set(Xr_select_i) & set(set(Xr_others_i))) == 0
        return Xr_select_i, idx_max_diff_arr, Xr_others_i, ctm_max_diff_arr, C_select_i_map, max_size
