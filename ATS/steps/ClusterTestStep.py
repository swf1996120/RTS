import numpy as np

from ATS import ats_config


# 1. 测试数据聚类
class ClusterTestStep(object):
    # 将数据分割成 大于上边界的,小于下边界的,在边界之间的
    # 在边界间的是重要的
    # 带标号
    def split_data_region_with_idx(self, Tx_prob_matrixc, i, idx):
        Tx_i_prob_vec = Tx_prob_matrixc[:, i]  # 原始模型预测向量矩阵对lablei的预测

        # 小于0.5则该用例可能预测错的 错误:0.1 0.9 正确: 0.4 0.3 0.2
        S1_i = Tx_prob_matrixc[Tx_i_prob_vec < ats_config.boundary]  # 错误集
        idx_1 = idx[Tx_i_prob_vec < ats_config.boundary]
        # 大于0.5一定是对的,并且要小于0.99, 置信度太高的点1没有意义2会导致数值错误
        # 每一行是一个X_k
        S0_i = Tx_prob_matrixc[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]
        idx_0 = idx[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]

        # 置信度特别高的点待定
        S2_i = Tx_prob_matrixc[(Tx_i_prob_vec > ats_config.up_boundary)]
        idx_2 = idx[(Tx_i_prob_vec > ats_config.up_boundary)]

        return S2_i, idx_2, S0_i, idx_0, S1_i, idx_1

    # 不带编号
    def split_data_region(self, Tx_prob_matrixc, i):
        Tx_i_prob_vec = Tx_prob_matrixc[:, i]  # 原始模型预测向量矩阵对lablei的预测

        # 小于0.5则该用例可能预测错的 错误:0.1 0.9 正确: 0.4 0.3 0.2
        S1_i = Tx_prob_matrixc[Tx_i_prob_vec < ats_config.boundary]  # 错误集
        # 大于0.5一定是对的,并且要小于0.99, 置信度太高的点1没有意义2会导致数值错误
        # 每一行是一个X_k
        S0_i = Tx_prob_matrixc[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]

        # 置信度特别高的点待定
        S2_i = Tx_prob_matrixc[(Tx_i_prob_vec > ats_config.up_boundary)]
        return S2_i, S0_i, S1_i
