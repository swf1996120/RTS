import numpy as np

from ATS import ats_config


# 2 project投影和extend延长
class ProjectExtendStep(object):

    # X 矩阵,
    # 如果该函数计算量大,mars可并行化
    def get_projection_matrixc(self, X, p, q, n, i):
        x_k_dot_matrixc = []
        for x_k in X:
            x_k_dot = self.get_projection_point(i, p, q, n, x_k)  # 获取投影点
            x_k_dot_matrixc.append(x_k_dot)
        return np.array(x_k_dot_matrixc)

    # 求x_k在i,p,q平面的投影点x_k'
    def get_projection_point(self, i, p, q, n, A):
        one_third = 1 / 3
        two_third = 2 / 3
        A_dot = np.zeros(A.shape)
        A_dot[i] = two_third * A[i] - one_third * A[p] - one_third * A[q] + one_third
        A_dot[p] = two_third * A[p] - one_third * A[q] - one_third * A[i] + one_third
        A_dot[p] = two_third * A[q] - one_third * A[p] - one_third * A[i] + one_third
        return A_dot

    # 2.extend 延长
    # 获取距离
    # 可并行化
    def get_i_distance_list(self, X, i):
        i_distance_list = []
        for x_k_dot in X:
            d = ats_config.up_boundary - x_k_dot[i]  # 不用1减
            i_distance_list.append(d)
        return i_distance_list

        #  可并行化
        # 延长到交线

    def extend_line(self, X, i):
        x_k_dot_dot_matrixc = X.copy()  # 复制一份矩阵
        n = len(x_k_dot_dot_matrixc[0])  # i的范围 [0,n)
        for x_k_dot in x_k_dot_dot_matrixc:  # 遍历每一个X
            d = 1 - x_k_dot[i]  # 计算 d
            for j in range(n):
                if j == i:
                    x_k_dot[j] = ats_config.boundary
                    continue
                else:
                    x_k_dot[j] = ((1 - ats_config.boundary) / d) * x_k_dot[j]
                    # 计算覆盖
        return x_k_dot_dot_matrixc

    # 求x_k在i, p, q平面的投影点x_k
    # x_k'是n维的,后续只用ipq这三个维度
    # def get_projection_point(i, p, q, n, A):
    #     # 顶点i
    #     P_i = get_vertex(i, n)
    #     # 法向量n
    #     n = get_normal_vector(i, p, q, n)
    #     # 向量P_iA
    #     P_iA = A - P_i
    #     # 投影向量P_iA_dot
    #     P_iA_dot = get_proj(P_iA, n)
    #     # 投影点坐标
    #     A_dot = P_i + P_iA_dot
    #     return A_dot

    # ##################################################################### 工具函数
    #
    # # u在v上投影
    # def get_proj(u, v):
    #     v_norm = np.sqrt(sum(v ** 2))
    #     proj_of_u_on_v = (np.dot(u, v) / v_norm ** 2) * v
    #     return u - proj_of_u_on_v
    #
    # # 获取平面顶点 顶点:第i个为1 ,其余为0
    # # i : [0,n)
    # def get_vertex(i, n):
    #     vec = np.zeros((n))
    #     vec[i] = 1
    #     return vec
    #
    # # 获取平面法向量
    # def get_normal_vector(i, p, q, n, ):
    #     normal_vec = np.zeros((n))
    #     normal_vec[i] = 1
    #     normal_vec[p] = 1
    #     normal_vec[q] = 1
    #     return normal_vec
