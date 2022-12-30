from ATS.steps.ClusterTestStep import ClusterTestStep
from ATS.steps.PatternFitnessStep import PatternFitnessStep
from ATS.steps.PatternGatherStep import PatternGatherStep
from ATS.steps.ProjectExtendStep import ProjectExtendStep
from ATS.steps.Utils import ATSUtils


# 在数据选择前的
# 算法步骤
class Steps(object):
    def __init__(self):
        self.cluster_test_step = ClusterTestStep()  # 测试数据聚类
        self.project_extend_step = ProjectExtendStep()  # 投影和延长
        self.pattern_gather_step = PatternGatherStep()  # 错误模式收集
        self.pattern_fitness_step = PatternFitnessStep()  # 错误模式度量计算
        self.ats_utils = ATSUtils()  # 工具类

    # 获取覆盖对
    def get_ck_list_map(self, S0_i, n, i):
        pq_list = self.ats_utils.get_p_q_list(n, i)  # 获得pq点集
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
