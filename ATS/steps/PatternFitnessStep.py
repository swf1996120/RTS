import numpy as np

# 4. 错误模式计算
from ATS import ats_config
from ATS.steps.Utils import ATSUtils


class PatternFitnessStep(object):

    def __init__(self):
        self.ats_utils = ATSUtils()

    # 将新的覆盖对放入SelectMap中
    def union_cov_maps(self, Cx, Cr_select_i_map, X_no_i):
        res_map = {}
        for x_pq in Cx:
            insert_falg = True
            [p, q, a_ins, b_ins, _] = x_pq
            CK_pq = Cr_select_i_map["{}_{}".format(p, q)].copy()
            # 如果CK_pq为空,则直接插入
            if len(CK_pq) == 0:
                res_map["{}_{}".format(p, q)] = CK_pq
            else:
                for ix in range(len(CK_pq)):
                    [_, _, a_cur, b_cur, _] = CK_pq[ix]
                    # 如果改点的左端点比现有的左端点大,就继续向后找
                    if a_ins > a_cur:
                        continue
                    # 找到第一个比改点小或者等于的点
                    else:
                        CK_pq.insert(ix, x_pq)
                        insert_falg = False
                        break
            if insert_falg:
                CK_pq.append(x_pq)
            res_map["{}_{}".format(p, q)] = CK_pq
        return res_map

    # 计算覆盖长度
    def get_cov_length_map(self, ck_list_map, n, i, ):
        s_pq_arr = []
        pq_list = self.ats_utils.get_p_q_list(n, i)
        for (p, q) in pq_list:
            # 统计并集
            ck_list = ck_list_map["{}_{}".format(p, q)]
            Ck_pq = self.statistic_union(p, q, ck_list)
            # 计算覆盖长度s(Ck_pq)
            s_pq = self.get_cov_length(Ck_pq)
            s_pq_arr.append(s_pq)
            # 计算总覆盖长度
        return s_pq_arr

    # 统计某个ipq并集,返回一个list
    def statistic_union(self, p, q, Ck_pq_temp, sort=True):
        if len(Ck_pq_temp) == 0:
            return 0
        # 计算 Ck_pq
        Ck_pq = Ck_pq_temp.copy()
        if sort:
            Ck_pq.sort(key=lambda x: (x[2]))  # 即按照a的大小顺序排序
        res = []
        s_pre = Ck_pq[0][2]  # 上一个线段的开头
        e_pre = Ck_pq[0][3]  # 上一个线段的末尾
        for i in range(1, len(Ck_pq)):
            s_cur = Ck_pq[i][2]  # 当前线段的开头
            e_cur = Ck_pq[i][3]  # 当前线段的末尾
            if s_cur <= e_pre:  # 如果当前线段的开头小于上一个线段的末尾
                # 合并两个线段
                e_pre = max(e_cur, e_pre)  # 将两个线段中更长的末尾更新
            else:
                # 出现了一个新的线段
                res.append([p, q, s_pre, e_pre, e_pre - s_pre])  # 将原有线段添加到结果
                s_pre = s_cur
                e_pre = e_cur
        res.append([p, q, s_pre, e_pre, e_pre - s_pre])
        return res

    # 计算覆盖长度
    def get_cov_length(self, Ck_pq):
        total_length = 0
        for i in range(len(Ck_pq)):
            total_length += Ck_pq[i][3] - Ck_pq[i][2]
        return total_length

    # 计算覆盖对map中所覆盖的总长度
    def get_cov_pair_map_len(self, Ck_pq_map, n, i):
        l_total = 0
        for (p, q) in self.ats_utils.get_p_q_list(n, i):
            CK_pq = Ck_pq_map["{}_{}".format(p, q)]
            if len(CK_pq) != 0:
                l = np.array(CK_pq)[:, 4].sum()
                l_total += l
        return l_total

    # 统计所有ipq的并集,返回一个map
    def statistic_union_map(self, Ck_pq_map, n, i):
        res_map = {}
        for (p, q) in self.ats_utils.get_p_q_list(n, i):
            key = "{}_{}".format(p, q)
            CK_pq = Ck_pq_map[key]
            CK_pq = self.statistic_union(p, q, CK_pq, sort=False)  # 不排序
            res_map[key] = CK_pq
        return res_map

    # 给长度计算覆盖率
    def get_cov_c(self, s, n):
        c = s / ((1 - ats_config.boundary) * (self.get_cn2(n)))
        return c

    # 计算c_n-1^2
    def get_cn2(self, n):
        return (1 / 2 * (n - 1) * (n - 2))

    # 计算覆盖长度和覆盖率
    def get_cov_s_and_c(self, s_pq_arr, n):
        # 计算总覆盖长度
        s = np.array(s_pq_arr).sum()
        # 计算覆盖率
        c = self.get_cov_c(s, n)
        return s, c
