import os
import time

from ATS.steps.Steps import Steps
from utils.utils import get_data_by_label
import pandas as pd
import numpy as np


class ATSmeasure(Steps):

    # 带覆盖率的方差
    def cal_d_v(self, Tx, Ty, n, M, base_path=None, is_anlyze=False, suffix=""):
        c, v, s = self.cal_d_detail(Tx, Ty, n, M, base_path=base_path, is_anlyze=is_anlyze, suffix=suffix)
        return c, v

    # 适配
    def cal_d(self, Tx, Ty, n, M, base_path=None, is_anlyze=False):
        c, v, s = self.cal_d_detail(Tx, Ty, n, M, base_path=base_path, is_anlyze=is_anlyze)
        return c

    # 计算div覆盖率
    def cal_d_detail(self, Tx, Ty, n, M, base_path=None, is_anlyze=False, suffix=""):
        df = None
        c_arr = []  # 存储覆盖率
        S1 = []  # 错误集
        for i in range(n):
            # print("i", i)
            csv_data = {}
            Tx_i, Ty_i = get_data_by_label(Tx, Ty, i)
            if Tx_i.size == 0:
                # 没有数据则覆盖率为0
                c_arr.append(0)
            else:
                Tx_prob_matrixc = M.predict(Tx_i)  # 原始预测向量矩阵
                # 选定区域为S_mid
                ##########################################
                S_up, S_mid, S_low = self.cluster_test_step.split_data_region(Tx_prob_matrixc, i)
                S1.append(S_low)
                # print("len(S0_i)", len(S_mid))
                if S_mid.size == 0:  # 目标分块里没有数据
                    c_arr.append(0)
                    continue
                # 按i p q 获取覆盖对 ipq a b
                ss = time.time()
                ck_list_map = self.get_ck_list_map(S_mid, n, i)
                ee = time.time()
                # 覆盖对分析
                # if is_anlyze:
                #     self.ck_pq_analyze(ck_list_map, n, i, base_path, S_mid, )
                sss = time.time()
                # 计算覆盖长度与覆盖率率
                # 计算覆盖长度
                s_pq_arr = self.pattern_fitness_step.get_cov_length_map(ck_list_map, n, i)
                # 计算第i个lable总覆盖长度与覆盖率率
                s, c_i = self.pattern_fitness_step.get_cov_s_and_c(s_pq_arr, n)
                # print("覆盖率 ", c_i)
                c_arr.append(c_i)
                eee = time.time()
                # print("cal cov over ..")

                csv_data["label"] = i
                csv_data["数据总量"] = len(Tx_i)
                csv_data["S_up大小"] = len(S_up)
                csv_data["S_low大小"] = len(S_low)
                csv_data["S_mid大小"] = len(S_mid)
                csv_data["covpair_time"] = ee - ss
                csv_data["union_time"] = eee - sss
                csv_data["cov_len"] = s
                csv_data["cov"] = c_i
                if df is None:  # 如果是空的
                    df = pd.DataFrame(csv_data, index=[0])
                else:
                    df.loc[df.shape[0]] = csv_data
        c = np.array(c_arr).mean()  # 整体覆盖率
        v = np.array(c_arr).var()

        base_path = base_path + "/temp_res"
        os.makedirs(base_path, exist_ok=True)
        csv_path = base_path + "/" + "{}_profile_output.csv".format(suffix)
        if df is not None:
            df.to_csv(csv_path, index=False)
        if len(S1) == 0:
            len_S1 = 0
        else:
            len_S1 = len(np.concatenate(S1, axis=0))
        return c, v, len_S1
