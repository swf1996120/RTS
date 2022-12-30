import logging
import os

import imagehash
import joblib
import torch
from keras_preprocessing.image import array_to_img
from scipy.special import softmax, logsumexp
from sklearn.linear_model import LogisticRegressionCV
from torch.backends import cudnn

import BestSolution1
import BestSolution_filter
import BestSolution_var
from ATS.ATS import ATS
from BestSolution import obtain_ssim, conquer, ob_sus_correct, get_noise_threshold
from selection_method.necov_method import metrics
import numpy as np

from selection_method.rank_method.CES.condition import CES_ranker
from utils import model_conf


class Ranker(object):
    def __init__(self, model, x):
        self.model = model
        self.x = x

    # gini系数
    def gini_rank(self):
        pred_test_prob = self.model.predict(self.x)
        gini_rank = metrics.deep_metric(pred_test_prob)
        return gini_rank

    def dac_rank_va1(self, nb_classes, dataname, groups=100, x_train=None, y_train=None, sim_number=5):

        select_pro = self.model.predict(self.x)
        select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签
        train_pro = self.model.predict(x_train)

        x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in self.x])
        x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])

        ssim_datas = obtain_ssim(data_name=dataname, sim_tolerate_num=sim_number, x_select=self.x, x_train=x_train,
                                 y_train=y_train,
                                 x_select_hash=x_select_hash,
                                 x_train_hash=x_train_hash)

        suspicious_array, correct_array = BestSolution_var.ob_sus_correct_var1(sim_tolerate_num=sim_number,
                                                                               ssim_datas=ssim_datas,
                                                                               select_pro=select_pro,
                                                                               select_lable=select_lable,
                                                                               train_pro=train_pro,
                                                                               y_train=y_train)

        temp = [suspicious_array, correct_array]

        rank_list = []

        for indexs in temp:
            # 获取的是索引
            scores = conquer(select_lable, select_pro, indexs, groups, nb_classes)
            # 获取索引+不确定度
            rank_list += scores

        rank_list = np.array(rank_list).astype(np.int)

        return rank_list

    def dac_rank_va2(self, nb_classes, dataname, groups=100, x_train=None, y_train=None, sim_number=5):

        select_pro = self.model.predict(self.x)
        select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签

        x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in self.x])
        x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])

        noise_threshold = get_noise_threshold(data_name=dataname, sim_tolerate_num=sim_number, x_train=x_train,
                                              y_train=y_train, x_train_hash=x_train_hash)

        ssim_datas = obtain_ssim(data_name=dataname, sim_tolerate_num=sim_number, x_select=self.x, x_train=x_train,
                                 y_train=y_train,
                                 x_select_hash=x_select_hash,
                                 x_train_hash=x_train_hash)

        suspicious_array, noise_array = BestSolution_var.ob_sus_correct_var2(ssim_datas=ssim_datas,
                                                                                            select_pro=select_pro,
                                                                                            noise_threshold=noise_threshold)

        temp = [suspicious_array, noise_array]

        rank_list = []

        for indexs in temp:
            # 获取的是索引
            scores = conquer(select_lable, select_pro, indexs, groups, nb_classes)
            # 获取索引+不确定度
            rank_list += scores

        rank_list = np.array(rank_list).astype(np.int)

        return rank_list

    def dac_rank_va3(self, nb_classes, dataname, groups=100, x_train=None, y_train=None, sim_number=5):

        select_pro = self.model.predict(self.x)
        select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签
        train_pro = self.model.predict(x_train)

        x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in self.x])
        x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])

        noise_threshold = get_noise_threshold(data_name=dataname, sim_tolerate_num=sim_number, x_train=x_train,
                                              y_train=y_train, x_train_hash=x_train_hash)

        ssim_datas = obtain_ssim(data_name=dataname, sim_tolerate_num=sim_number, x_select=self.x, x_train=x_train,
                                 y_train=y_train,
                                 x_select_hash=x_select_hash,
                                 x_train_hash=x_train_hash)

        suspicious_array, noise_array, correct_array = ob_sus_correct(sim_tolerate_num=sim_number,
                                                                      ssim_datas=ssim_datas,
                                                                      select_pro=select_pro, select_lable=select_lable,
                                                                      train_pro=train_pro, y_train=y_train,
                                                                      noise_threshold=noise_threshold,
                                                                      filter_noise=False)

        temp = [suspicious_array, noise_array, correct_array]

        rank_list = []

        for indexs in temp:
            # 获取的是索引
            scores = BestSolution_var.conquer(indexs=indexs)
            # 获取索引+不确定度
            rank_list += scores

        rank_list = np.array(rank_list).astype(np.int)

        return rank_list

    def dac_rank(self, nb_classes, dataname, groups=100, x_train=None, y_train=None, sim_number=5):

        select_pro = self.model.predict(self.x)
        select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签
        train_pro = self.model.predict(x_train)

        x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in self.x])
        x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])

        noise_threshold = get_noise_threshold(data_name=dataname, sim_tolerate_num=sim_number, x_train=x_train,
                                              y_train=y_train, x_train_hash=x_train_hash)

        ssim_datas = obtain_ssim(data_name=dataname, sim_tolerate_num=sim_number, x_select=self.x, x_train=x_train,
                                 y_train=y_train,
                                 x_select_hash=x_select_hash,
                                 x_train_hash=x_train_hash)

        suspicious_array, noise_array, correct_array = ob_sus_correct(sim_tolerate_num=sim_number,
                                                                      ssim_datas=ssim_datas,
                                                                      select_pro=select_pro, select_lable=select_lable,
                                                                      train_pro=train_pro, y_train=y_train,
                                                                      noise_threshold=noise_threshold,
                                                                      filter_noise=False)

        temp = [suspicious_array, noise_array, correct_array]

        rank_list = []

        for indexs in temp:
            # 获取的是索引
            scores = conquer(select_lable, select_pro, indexs, groups, nb_classes)
            # 获取索引+不确定度
            rank_list += scores

        rank_list = np.array(rank_list).astype(np.int)

        return rank_list

    def dac_filter_rank(self, nb_classes, dataname, groups=100, x_train=None, y_train=None, sim_number=5):
        select_pro = self.model.predict(self.x)
        select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签
        train_pro = self.model.predict(x_train)

        x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in self.x])
        x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])

        noise_threshold = BestSolution_filter.get_noise_threshold(data_name=dataname, sim_tolerate_num=sim_number,
                                                                  x_train=x_train,
                                                                  y_train=y_train, x_train_hash=x_train_hash)

        ssim_datas = BestSolution_filter.obtain_ssim(data_name=dataname, sim_tolerate_num=sim_number, x_select=self.x,
                                                     x_train=x_train,
                                                     y_train=y_train,
                                                     x_select_hash=x_select_hash,
                                                     x_train_hash=x_train_hash)

        suspicious_array, noise_array, correct_array = BestSolution_filter.ob_sus_correct(sim_tolerate_num=sim_number,
                                                                                          ssim_datas=ssim_datas,
                                                                                          select_pro=select_pro,
                                                                                          select_lable=select_lable,
                                                                                          train_pro=train_pro,
                                                                                          y_train=y_train,
                                                                                          noise_threshold=noise_threshold)

        temp = [suspicious_array, noise_array, correct_array]

        rank_list = []

        for indexs in temp:
            # 获取的是索引
            scores = BestSolution_filter.conquer(select_lable, select_pro, indexs, groups, nb_classes)
            # 获取索引+不确定度
            rank_list += scores

        rank_list = np.array(rank_list).astype(np.int)

        return rank_list

    def dac_2_rank(self, nb_classes, dataname, groups=100, x_train=None, y_train=None, sim_number=5, select_size=4000):
        select_pro = self.model.predict(self.x)
        select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签
        train_pro = self.model.predict(x_train)

        x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in self.x])
        x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])

        noise_threshold = BestSolution1.get_noise_threshold(data_name=dataname, sim_tolerate_num=sim_number,
                                                            x_train=x_train,
                                                            y_train=y_train, x_train_hash=x_train_hash)

        ssim_datas = BestSolution1.obtain_ssim(data_name=dataname, sim_tolerate_num=sim_number, x_select=self.x,
                                               x_train=x_train,
                                               y_train=y_train,
                                               x_select_hash=x_select_hash,
                                               x_train_hash=x_train_hash)

        suspicious_array, noise_array, correct_array = BestSolution1.ob_sus_correct(sim_tolerate_num=sim_number,
                                                                                    ssim_datas=ssim_datas,
                                                                                    select_pro=select_pro,
                                                                                    select_lable=select_lable,
                                                                                    train_pro=train_pro,
                                                                                    y_train=y_train,
                                                                                    noise_threshold=noise_threshold,
                                                                                    filter_noise=True)

        temp = [suspicious_array, noise_array, correct_array]

        rank_list = []

        for indexs in temp:
            # 获取的是索引
            scores, terminal = BestSolution1.conquer(select_lable, select_pro, indexs, groups, nb_classes, wise=2,
                                                     select_size=select_size)
            # 获取索引+不确定度
            rank_list += scores

            if terminal:
                break

        assert len(rank_list) >= select_size

        rank_list = np.array(rank_list).astype(np.int)

        return rank_list

    # 最大概率
    def max_p_rank(self):
        pred_test_prob = self.model.predict(self.x)
        metrics = np.max(pred_test_prob, axis=1)
        max_p_rank = np.argsort(metrics)  # 有小到大
        return max_p_rank

    # 敏感性分析
    # pd pv pe
    def noise_rank(self):
        pred_test_prob = self.model.predict(self.x)
        matrix = np.sort(pred_test_prob, axis=1)[:, ::-1]
        pd_score_arr = []
        for p in matrix:
            sum = 0
            for i in range(len(p) - 1):
                diff = p[i] - p[i + 1]
                assert diff >= 0
                sum += diff
            pd_score = sum / len(p)
            pd_score_arr.append(pd_score)
        pd_score_rank = np.argsort(pd_score_arr)  # 由小到大
        return pd_score_rank

    def div_rank(self, th=0.001, nb_classes=None):
        ats = ATS()
        pred_test_prob = self.model.predict(self.x)
        Y_psedu_select = np.argmax(pred_test_prob, axis=1)
        div_rank, _, _ = ats.get_priority_sequence(self.x, Y_psedu_select, nb_classes, self.model, th=0.001)

        return div_rank

    def div_rank_ctm(self):
        ...

    #     ats = ATS()
    #     pred_test_prob = self.model.predict(self.x)
    #     Y_psedu_select = np.argmax(pred_test_prob, axis=1)
    #     div_rank = ats.get_ctm_priority_sequence(self.x, Y_psedu_select, model_conf.fig_nb_classes,
    #                                              self.model,
    #                                              base_path=None, use_add=True, prefix=None,
    #                                              is_save_ps=True)
    #
    #     assert len(div_rank) == len(self.x)
    #     assert len(div_rank) == len(set(div_rank))
    #
    #     return div_rank

    def ces_rank(self, select_size=None):
        if select_size is None:
            select_size = len(self.x)
        print(f"ces_rank size is {select_size}")
        return CES_ranker().run(self.model, self.x, select_size)

    def lsa_rank(self, cov_initer, y_test=None):
        if y_test is None:
            pred_test_prob = self.model.predict(self.x)
            Y_psedu_select = np.argmax(pred_test_prob, axis=1)
            y_test = Y_psedu_select
        lsc = cov_initer.get_lsc()
        rate = lsc.fit(self.x, y_test)
        rank_lst = lsc.rank_2()
        return np.array(rank_lst)
        # lsc dsc中的排序使用了fit中的中间结果,因此时间成本要加入进去
        #   cov_initer = get_cov_initer(x_train, y_train, data_name, model_name)
        #         lsc = cov_initer.get_lsc()

    def random_rank(self):
        return np.random.permutation(len(self.x))

    # http://www.github.com/bntejn/keras-prioritizer
    def bayesian_rank(self):
        ...

    # pace
    # https://github.com/pace2019/pace
    def pace_rank(self):
        ...

    # var
    # Test Selection for Deep Learning Systems
    def var_rank(self):
        ...
