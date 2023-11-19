import bisect
import os
from collections import defaultdict

from ATS.selection.AbsSelectStrategy import AbsSelectStrategy
from utils.utils import get_data_by_label_with_idx
import numpy as np
import pandas as pd


# 每个psedu-label 尽可能的均匀的选
# 只能给定数量来选取,选取的数据为一个集合,集合中数据无顺序
# 弃用 @deprecated
class UniformSelectStrategy(AbsSelectStrategy):

    def assert_Tx(self, Tx_i, target_size):
        if Tx_i.size == 0:
            raise ValueError("该lable下没有数据")  # 应该避免
        if len(Tx_i) < target_size:
            raise ValueError("该lable下的数据不够选")  # 应该避免

    def assert_S_mid(self, S_mid, target_size, symbol, idx_up, idx_mid):
        select_lable = None
        select_idx = []
        s_mid_len = len(S_mid)
        idx_mid = list(idx_mid)
        if s_mid_len == target_size:  # 目标分块里就这么多数据
            select_lable = symbol.SELECT_ALL_MID_LABLES
            select_idx = idx_mid
        elif s_mid_len < target_size:  # 数据不够了,S_mid全选,再从S_up里选点过来
            select_lable = symbol.SELECT_ALL_MID_LABLES_WITH_HIGH
            select_idx += idx_mid
            diff_size = target_size - s_mid_len
            idx_up_ran = list(np.random.randint(len(idx_up), size=diff_size))  # 随机选取idx行
            select_idx += idx_up_ran
            if len(select_idx) != target_size:
                raise ValueError("do not slect target_size points from S_up ")  # 没有从S_up里选出正确个数的点"
        elif s_mid_len == 0:  # 目标分块里没有数据,全从S_up里选
            select_lable = symbol.SELECT_ZERO_LABLES
            diff_size = target_size - s_mid_len
            idx_up_ran = list(np.random.randint(len(idx_up), size=diff_size))  # 随机选取idx行
            select_idx += idx_up_ran
            raise ValueError("taget region does not have points")  # 实验中如果分块里没有数据就抛异常 该分块下没有数据
        return select_lable, select_idx

    # 数据选择算法
    # 适配各种情况(不是所有情况都会执行数据选择算法)
    def datasets_select(self, Tx, Ty, n, M, ori_target_size, extra_size=0, base_path=None, is_analyze=False):
        print(ori_target_size)
        df = None
        csv_path = base_path + "/data_select_{}.csv".format(ori_target_size)
        csv_data = {}
        c_arr = []  # 存储覆盖率
        max_size_arr = []
        Xr_select = []
        select_lable_arr = []
        symbol = select_status()
        for i in range(n):
            if extra_size == 0:
                target_size = ori_target_size
            else:
                target_size = ori_target_size + 1
                extra_size -= 1
            C_select_i_map = None
            max_cov_point_size = -1  # 最大覆盖点个数
            # 返回值,返回一个选择标记和选择数据编号集
            print("i", i)
            csv_data["label"] = i
            Tx_i, Ty_i, T_idx_arr = get_data_by_label_with_idx(Tx, Ty, i)  # 返回所有数据的绝对下标

            self.assert_Tx(Tx_i, target_size)  # 异常流程,标签不够
            Tx_prob_matrixc = M.predict(Tx_i)  # 原始预测向量矩阵
            if len(Tx_i) == target_size:  # 刚刚好
                select_lable = symbol.SELECT_ALL_LABLES
                abs_idx = T_idx_arr
                C_select_i_map = self.get_ck_list_map(Tx_prob_matrixc, n, i)  # 改成Tx_prob_matrixc
            else:  # 算法流程
                # 选定区域为S_mid
                ##########################################
                # 未分块, 通常S_up为置信度>0.99的,S_mid为置信度<0.99的,S_low为空
                # 返回T_idx_arr的相对下标
                S_up, rel_idx_up, S_mid, rel_idx_mid, S_low, rel_idx_low \
                    = self.cluster_test_step.split_data_region_with_idx(Tx_prob_matrixc, i,
                                                                        np.array(range(len(Tx_prob_matrixc))))

                print("len(S0_i)", len(S_mid))
                # 异常流程2,S_mid不够
                # 返回T_idx_arr的相对下标
                select_lable, rel_select_idx = self.assert_S_mid(S_mid, target_size, symbol, rel_idx_up, rel_idx_mid)
                # 正常流程继续
                if select_lable is None:  # 只有运行算法,才能有最大覆盖
                    select_lable, rel_select_idx, C_select_i_map, max_cov_point_size \
                        = self.datasets_select_detail(symbol, S_mid, n, i, target_size, rel_idx_mid,
                                                      base_path=base_path,
                                                      is_analyze=is_analyze)
                    # print(select_lable, )
                    # print(rel_select_idx)
                    # # print(C_select_i_map)
                    # print(max_cov_point_size)
                if C_select_i_map is None:  # 为None 代表补点了
                    C_select_i_map = self.get_ck_list_map(Tx_prob_matrixc[rel_select_idx], n, i)  # 根据T_idx_arr的相对下标拿数据
                abs_idx = T_idx_arr[rel_select_idx]  # 获得绝对下标
            # 计算所选数据集的覆盖率,与覆盖长度
            s_pq_arr = self.pattern_fitness_step.get_cov_length_map(C_select_i_map, n, i, )
            # 计算第i个lable总覆盖长度与覆盖率率
            s, c_i = self.pattern_fitness_step.get_cov_s_and_c(s_pq_arr, n)
            print("覆盖率 ", c_i)
            # 添加结果
            Xr_select.append(abs_idx)  # 所选数据编号
            select_lable_arr.append(select_lable)  # 该lable下的状态
            csv_data["select_lable"] = select_lable
            c_arr.append(c_i)  # 该lable的覆盖率
            csv_data["div"] = c_i
            # 最大覆盖个数:
            max_size_arr.append(max_cov_point_size)
            csv_data["max_cov"] = max_cov_point_size
            if df is None:  # 如果是空的
                df = pd.DataFrame(csv_data, index=[0])
            else:
                df.loc[df.shape[0]] = csv_data
            df.to_csv(csv_path, index=False)
        return Xr_select, select_lable_arr, c_arr, max_size_arr

    # 分析覆盖率增长变化
    # 排序表
    def datasets_select_detail(self, symbol, S_mid, n, i, target_size, idx_mid, base_path=None, is_analyze=False):
        temp_c_arr = []
        max_cov_point_size = target_size  # 最大覆盖个数
        rel_mid_idx = []  # 使用idx_mid的相对编号
        C_select_i_map = defaultdict(list)
        X_no_i = []
        select_lable = symbol.SELECT_MID_LABLES_CAM
        # 按i p q 获取覆盖对 ipq a b
        ck_list_map = self.get_ck_list_map(S_mid, n, i)
        ### 将 get_ck_list_map reshpe一下
        C_Xr_list = list(map(list, zip(*ck_list_map.values())))  # 行smaples 列pq_nums 每行为该用例x对应的在所有pq维度下的覆盖对
        if len(idx_mid) != len(C_Xr_list):
            raise ValueError("len idx not eq len data")  # 下标个数与数据个数不符
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
        # if is_analyze:
        #     df = pd.DataFrame({"idx": all_rank_idx, "len": all_rank_cov_len})
        #     df.to_csv(base_path + "/{}_data_rank.csv".format(i))
        #     return
        # # 4.将原有的数据按照覆盖长度的大小重新排序
        # C_Xr_list = np.array(C_Xr_list)[all_rank_idx]
        # # 5. 将第一个点加入到集合中,开始算法
        # rel_mid_idx.append(all_rank_idx[0])  # 添加该数据编号
        # X_no_i.append(all_rank_idx[0])
        # 点选择
        while len(rel_mid_idx) < target_size:
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
                if abs(s_diff) <= 0.001:  # 浮点数计算,如果没有大于一定阈值,认为是没变化
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
                # print("# max_s_i", max_idx)
                rel_mid_idx.append(max_idx)  # 添加该数据编号
                X_no_i.append(max_idx)  # 该数据被选过
                # print("# X_no_i", len(X_no_i))
                # print(C_select_i_map)
                C_select_i_map = max_Cr_select_i_map.copy()  # 更新添加后的覆盖对集合
                all_rank_idx = all_rank_idx_copy.copy()
                all_rank_cov_len = all_rank_cov_len_copy.copy()
                if max_s_diff < 0.005:
                    pass
            else:  # 没找到一个新的点,则循环结束
                if len(X_no_i) != len(S_mid):
                    # 已经无法选择点了,但X_no_i没有添加所有的编号
                    raise ValueError("no point left but X_no_i does not hav all data points")
                if len(rel_mid_idx) == target_size:  # 刚好选择完了所有点
                    select_lable = symbol.SELECT_MID_LABLES_CAM_ALL
                    max_cov_point_size = len(rel_mid_idx)
                    break
                else:  # 补点
                    add_num = target_size - len(rel_mid_idx)
                    select_lable = symbol.SELECT_MID_LABLES_CAM_CTM
                    # 记录最大选择点的数量
                    max_cov_point_size = len(rel_mid_idx)
                    print("cam max ={}", format(len(rel_mid_idx)))
                    Xr_ctm_idx = list(set(X_no_i) - set(rel_mid_idx))  # 没选过,但又被淘汰了的数据编号
                    C_Xr_CTM_list = np.array(C_Xr_list)[Xr_ctm_idx]  # 剩下的这些数据里,选覆盖大的
                    # 排序 C_Xr_CTM_list, 把每行第4个数加起来后排序
                    sorted_arr = []
                    for i_pq, x_pq in zip(Xr_ctm_idx, C_Xr_CTM_list):
                        len_s = np.array(x_pq)[:, 4].sum()
                        sorted_arr.append([i_pq, len_s])
                    sorted_arr.sort(key=lambda x: (x[1]), reverse=True)  # 由大到小排序

                    select_ctm_x = np.array(sorted_arr)[:, 0].astype("int").tolist()
                    select_ctm_x = select_ctm_x[:add_num]  # 选择下标,与补点数
                    rel_mid_idx += select_ctm_x  # 合并至原始下标内
                    # print(rel_mid_idx)
                    break
        if is_analyze:
            csv_dir = base_path + "/data_select_profile/"
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = csv_dir + "{}.csv".format(i)
            df = pd.DataFrame(temp_c_arr)
            df.to_csv(csv_path)
        idx_mid = np.array(idx_mid)
        Xr_select_i = idx_mid[rel_mid_idx]  # 根据相对编号获得原始数据的编号
        # np.save(base_path + "/mid_rel_idx_{}.npy".format(i), S_mid[rel_mid_idx[-1]])
        # np.save(base_path + "/{}.npy".format(i), rel_mid_idx)
        if len(Xr_select_i) != target_size:
            raise ValueError("data points size not eq target_size")  # "没有选取到target_size个数的点"
        if len(Xr_select_i) != len(set(Xr_select_i)):
            raise ValueError("some data points  repeatly select")  # 有数据点被重复选取了
        return select_lable, Xr_select_i, C_select_i_map, max_cov_point_size


class select_status(object):

    def __init__(self) -> None:
        self.SELECT_ALL_LABLES = 0  # 选择了该标签下所有数据
        self.SELECT_ZERO_LABLES = 1  # 该标签有数据,但Smid中没有数据
        self.SELECT_ALL_MID_LABLES = 2  # 该标签有数据,Smid中数据刚好全部被选中
        self.SELECT_ALL_MID_LABLES_WITH_HIGH = 3  # 该标签有数据,Smid中数据全部被选中后也不够,需要补选置信度高的其他数据
        self.SELECT_MID_LABLES_CAM = 4  # 该标签有数据,且Smid中数据全部选中后没有达到最大覆盖
        self.SELECT_MID_LABLES_CAM_ALL = 5  # 该标签有数据,且Smid中数据全部选中后刚好达到最大覆盖
        self.SELECT_MID_LABLES_CAM_CTM = 6  # 该标签有数据,且Smid中数据没选完就达到了最大覆盖,需要使用CTM方式选取Smid中数据
