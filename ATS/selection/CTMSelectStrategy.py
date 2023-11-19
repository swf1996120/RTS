import os
from collections import defaultdict
import numpy as np
from ATS.selection.AbsSelectStrategy import AbsSelectStrategy
from utils.utils import get_data_by_label_with_idx


# 每个数据只计算第一次覆盖长度
# 不进行迭代Adaptive计算
# 根据该长度排序
# 弃用 @deprecated

class CTMSelectStrategy(AbsSelectStrategy):

    # ctm
    def get_ctm_priority_sequence_detail(self, S_mid, n, i, idx_mid, ):
        temp_c_arr = []
        rel_mid_idx = []  # 使用idx_mid的相对编号
        idx_max_diff_arr = []
        C_select_i_map = defaultdict(list)  # 如果添加初始的数据,一定要把这里改了
        X_no_i = []
        # 按i p q 获取覆盖对 ipq a b
        ck_list_map = self.pattern_fitness_step.get_ck_list_map(S_mid, n, i)
        ### 将 get_ck_list_map reshpe一下
        C_Xr_list = list(map(list, zip(*ck_list_map.values())))  # 行smaples 列pq_nums 每行为该用例x对应的在所有pq维度下的覆盖对
        if len(idx_mid) != len(C_Xr_list):
            raise ValueError("len idx not eq len data")  # "下标个数与数据个数不符"
        ### step1 记录Ctm表格
        # 1.记录每个用例的下标及其最大覆盖长度
        all_rank_arr = []
        for ii, xi in enumerate(C_Xr_list):
            l = np.array(xi)[:, 4].sum()
            all_rank_arr.append([ii, l])
        # 2. 对覆盖长度排序
        all_rank_arr.sort(key=lambda x: (x[1]), reverse=False)  # 由小到大排序
        # 3. 获得排序后的下标及覆盖长度
        rel_mid_idx = np.array(all_rank_arr)[:, 0].astype("int").tolist()
        all_rank_cov_len = np.array(all_rank_arr)[:, 1].tolist()
        all_rank_idx = idx_mid[rel_mid_idx]
        return all_rank_idx, all_rank_cov_len

    def get_ctm_priority_sequence(self, Tx, Ty, n, M, base_path=None, use_add=True, prefix=None, is_save_ps=False):
        csv_data = {}
        ps_path = base_path + "/ps"
        ps_path_all = base_path + "/ps_all"
        idx_others = []
        idx_dict = {}
        for i in range(n):
            csv_data["label"] = i
            # 返回值,返回一个选择标记和选择数据编号集
            Tx_i, Ty_i, T_idx_arr = get_data_by_label_with_idx(Tx, Ty, i)  # 返回所有数据的绝对下标
            Tx_prob_matrixc = M.predict(Tx_i)  # 原始预测向量矩阵
            # 返回T_idx_arr的相对下标
            S_up, rel_idx_up, S_mid, rel_idx_mid, S_low, rel_idx_low \
                = self.cluster_test_step.split_data_region_with_idx(Tx_prob_matrixc, i,
                                                                    np.array(range(len(Tx_prob_matrixc))))

            rel_select_idx, all_rank_cov_len = self.get_ctm_priority_sequence_detail(S_mid, n, i,
                                                                                     rel_idx_mid, )
            # assert len(rel_idx_mid) == len(rel_select_idx)
            # assert (np.array(sorted(rel_idx_mid)) == np.array(sorted(rel_select_idx))).all()
            # assert len(set(rel_idx_up) | (set(rel_idx_mid))) == len(rel_idx_up) + len(rel_idx_mid)
            idx_others.append(T_idx_arr[rel_idx_up])
            # 数据编号
            abs_idx = T_idx_arr[rel_select_idx]
            for ix, cov_len in zip(abs_idx, all_rank_cov_len):
                if ix in idx_dict.keys():
                    raise ValueError()
                idx_dict[ix] = cov_len

        sort_list_dict = sorted(idx_dict.items(), key=lambda x: x[1], reverse=True)  # 由大到小排序
        sort_ix, sort_len = zip(*sort_list_dict)
        assert sort_len[0] >= sort_len[1]
        assert len(sort_ix) == len(set(sort_ix))

        idx_others = np.concatenate(idx_others, axis=0)
        np.random.seed(0)
        idx_others = np.random.permutation(idx_others)
        # assert len(idx_others) == len(set(idx_others))
        Xr_select = np.concatenate([sort_ix, idx_others], axis=0)
        # assert len(Xr_select) == len(set(Xr_select))
        if is_save_ps:
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
            x_all_idx = Xr_select
            save_path2 = ps_path_all + "/{}all.npy".format(prefix)
            np.save(save_path2, x_all_idx)
        return Xr_select
