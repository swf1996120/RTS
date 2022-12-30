from ATS.steps.ClusterTestStep import ClusterTestStep
from ATS.steps.PatternGatherStep import PatternGatherStep
from ATS.steps.ProjectExtendStep import ProjectExtendStep


class ATSUtils(object):

    def __init__(self):
        self.cluster_test_step = ClusterTestStep()  # 测试数据聚类
        self.project_extend_step = ProjectExtendStep()  # 投影和延长
        self.pattern_gather_step = PatternGatherStep()  # 错误模式收集

    # 计算量大可约减
    # 如果这里选择全部c92计算量太复杂的话
    # 可以选择9个里最大的一个or两个，和剩下8个的组合
    def get_p_q_list(self, n, i):
        num_list = list(range(n))
        num_list.remove(i)
        import itertools
        pq_list = []
        # 抛掉一个点,剩下的排列组合选2个 C92
        for pq in itertools.combinations(num_list, 2):
            pq_list.append(pq)
        return pq_list
        # 获取覆盖对
    def get_ck_list_map(self, S0_i, n, i):
        pq_list = self.get_p_q_list(n, i)  # 获得pq点集
        ck_map = {}  # 储存覆盖对ck
        # 对x矩阵以方式处理
        for (p, q) in pq_list:
            S0_projection_matrixc = self.project_extend_step.get_projection_matrixc(S0_i, p, q, n, i)
            i_distance_list = self.project_extend_step.get_i_distance_list(S0_projection_matrixc, i)
            x_k_dot_dot_matrixc = self.project_extend_step.extend_line(S0_projection_matrixc, i)
            ck_i_list = self.pattern_gather_step.get_cov_pair(i_distance_list, x_k_dot_dot_matrixc, p, q)
            if len(ck_i_list) != 0:  # 浮点数导致的负数值会导致ck_i_list 没元素
                ck_map["{}_{}".format(p, q)] = ck_i_list
            if len(i_distance_list) == len(x_k_dot_dot_matrixc) == len(ck_i_list):
                pass
            else:
                raise ValueError("len ck list  not eq data size")  # ck list 长度不对,不等于用例长度
        return ck_map