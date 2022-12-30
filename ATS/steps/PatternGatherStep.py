from ATS import ats_config
import numpy as np


# 3. 获错误模式收集
class PatternGatherStep(object):

    # 计算覆盖对
    def get_cov_pair(self, i_distance_list, x_k_dot_dot_matrixc, p, q):
        ck_list = []
        for d, x_k_dot_dot in zip(i_distance_list, x_k_dot_dot_matrixc):
            if d < 0 or x_k_dot_dot[p] < 0:  # 浮点数负数问题
                ck = [p, q, 0, 0, 0]  # 添加加一个不影响覆盖率的值
                ck_list.append(ck)
                continue
            L = self.get_cov_radius(d)  # 覆盖半径
            a = x_k_dot_dot[p] - L  #
            b = x_k_dot_dot[p] + L
            # print("==", x_k_dot_dot[p], a, b, d, L)
            ##################
            # 对 a,b 进行小数点约束
            ##################
            a = np.round(a, ats_config.round_num)
            b = np.round(b, ats_config.round_num)
            if a < 0:
                a = 0
            if b > 1 - ats_config.boundary:
                b = 1 - ats_config.boundary
            if a > b:
                a = b
            ck = [p, q, a, b, b - a]  # p,q表示对应维度，a,b表示覆盖起始点
            ck_list.append(ck)
        return ck_list

    # 计算覆盖半径
    # 保证d越大l越大即可
    def get_cov_radius(self, d):
        if ats_config.is_log:
            l = ats_config.log_ratio * np.log1p(d)
        else:
            l = ats_config.linear_ratio * d
            if ats_config.is_radius_th:
                if d < ats_config.radius_th:
                    l = 0
        return l
