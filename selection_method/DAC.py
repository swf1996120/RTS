#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Q1mi"
# Date: 2022/1/15
import logging
from typing import List

import joblib
import pandas as pd
import numpy as np
from piqa import SSIM
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegressionCV
from tensorflow.python.keras.models import model_from_json

from collections import OrderedDict, Counter
import os

from scipy.special import logsumexp, softmax

from torch.backends import cudnn
from tqdm import tqdm

import models.fashion.convnet
from gen_data.Dau import Dau
from gen_data.Dop import DauOperator
from models.svhn.allconv import AllConvNet
from utils import model_conf

import torch


def get_intervals(max_min_interval, num_intervals):
    '''
    根据num_intervals数目 划分区间
    :param max_min_interval: 每个特征空间的 最大值 与 最小值
    :param num_intervals: 区间的划分个数
    :return: List[Tuple[length_interval, number_intervals]]
    '''

    def get_interval(length, num_interval):
        length_interval = length / num_interval  # 单个区间的长度
        return length_interval, [min + i for i in np.arange(0, length, length_interval)]

    result = []
    for intervals in max_min_interval:
        max, min = intervals[0], intervals[1]
        result.append(get_interval((max - min), num_intervals))
    return result


def give_index(max_min_intervals, intervals, sample, i, keep):
    '''
    给出单个样本的区间索引 数字越大概率越大
    :param max_min_intervals: 每个特征空间的范围  最大值  最小值
    :param intervals: @link get_intervals 得到每个特征的 划分区间
    :param sample: @ 要判断的样本
    :param i: 最大的预测概率索引
    :param keep: 是否保持最大的概率
    :return: 返回每个softmax的值的索引区间
    '''
    sample_index = np.zeros(len(sample))  # 暂时生成一个index数组
    for index in range(len(sample)):
        interval = intervals[index]  # 第i个特征的intervals
        interval_length, interval_index = interval[0], interval[1]  # 第i个特征范围 单位长度 以及索引位置
        for inner_index in range(len(interval_index)):  # 逐个查看索引区间
            limitation = max_min_intervals[index][1] if inner_index == len(interval_index) - 1 else interval_index[
                                                                                                        inner_index] + interval_length  # 如果 最后的一个索引
            if interval_index[inner_index] <= sample[index] <= limitation:
                sample_index[index] = inner_index

    if not keep:  # 不保留主概率 将其概率值变为0
        sample_index[int(i)] = 0
    return sample_index


def get_socres(count_indexs, num_parameter, pred_test_prob, ood_scores):
    '''
    直接统计位于前端区间的个数
    '''
    def provide_score(counter, num_intervals):
        '''
        根据某个计算公式，提供每个样本的得分
        :return:
        '''

        scores = np.ones((num_intervals))  # 生成权重大小
        sum = 0
        for key, value in counter.items():
            sum += key * value * scores[int(key)]
        return sum

    scores = [(provide_score(count_indexs[i], num_parameter), i, pred_test_prob[i], ood_scores[i])
              for i in
              range(len(count_indexs))]

    return scores


def get_socres_var(interval_indexs, pred_test_prob, ood_scores):
    '''
    统计区间的离散程度
    '''
    def provide_score(indexs):
        '''
        根据某个计算公式，提供每个样本的得分
        :return:
        '''
        sum = np.var(indexs)
        return sum

    scores = [(provide_score(interval_indexs[i]), i, pred_test_prob[i], ood_scores[i])
              for i in
              range(len(interval_indexs))]

    return scores


def group_by_scores(scores):
    '''
    根据分数进行了分组
    :param scores: [scores, index, array_points, ood_scores]
    :return:
    '''
    dic = OrderedDict()
    for score in scores:
        group = dic.get(score[0], list())
        if len(group) == 0:
            dic[score[0]] = group
        group.append(score[1:])
    return dic


def get_othermethic(metric, data):
    '''
    返回最大的不确定度
    :param data:
    :return:
    '''
    return metric(data)


def Step_1(pred_test, pred_test_prob, ood_scores, num_parameter):
    '''

    :param pred_test:
    :param pred_test_prob:
    :param ood_scores:
    :param num_parameter:
    :return:
    '''
    # 得到最大最小区间
    max_min_interval = [(np.max(single_batch), np.min(single_batch)) for single_batch in pred_test_prob.T]

    # 根据设置的区间个数 求区间个数
    intervals = get_intervals(max_min_interval,
                              num_parameter)  # length_interval, [min + i for i in np.arange(0, length, length_interval)]

    # 得到每个样本的区间索引，越大索引概率越大
    all_indexs = [give_index(max_min_interval, intervals, pred_test_prob[i], pred_test[i], False)
                  for i in range(len(pred_test_prob))]


    # 区间索引统计
    # count_indexs = [Counter(single_index) for single_index in all_indexs]
    #
    # # 根据某个计算公式得到分数并排序
    # return sorted(get_socres(count_indexs, num_parameter, pred_test_prob, ood_scores), key=lambda x: x[0], reverse=True)

    # 根据某个计算公式得到分数并排序
    return sorted(get_socres_var(all_indexs, pred_test_prob, ood_scores), key=lambda x: x[0], reverse=True)


def get_uncertainty(data):
    '''
    返回最大的不确定度
    :param data:
    :return:
    '''
    return 1.0 - np.max(data)


def Step_2(group_dict):
    '''
    根据第三个指标进行最后的排序
    :param group_dict:  key:score | value:[index, pred_test_prob, ood_score]
    :return: [(uncertain, index, ood_score),....,]
    '''
    # print(group_dict)
    final_result = []
    for _, values in group_dict.items():
        group_datas = [(get_othermethic(metric=get_uncertainty, data=value[1]), value[0], value[2]) for value in
                       values]  # 将该大的分组下的序列求得了一个最大不确定度
        group_datas = sorted(group_datas, key=lambda x: x[0], reverse=True)
        final_result = final_result + group_datas
    return final_result


def get_catogery(real_number, dirs_sims, num_sims, y_train):
    ''''
    根据最相近的top-vector 计算是否为可遗样本
    '''
    values, indices = torch.topk(torch.tensor(dirs_sims, dtype=torch.float64), k=int(num_sims) + 1, dim=-1,
                                 largest=False,
                                 sorted=True)
    indices = indices[:, 1:]

    no_equl_pairs, equal_pairs = [], []
    for line_number in range(len(indices)):
        line_number, best_sim_sample_indexs = line_number, indices[line_number]

        for best_sim in best_sim_sample_indexs:
            if real_number[line_number][-1] != y_train[best_sim]:  # 预测标签
                no_equl_pairs.append(line_number)
                break
        else:
            equal_pairs.append(line_number)
    return no_equl_pairs, equal_pairs


def Step_3(datas, neighbers, train_pred_prob, y_train, noise_idx):
    '''
    先进行一个初步的划分
    :return:
    '''
    # pred_probility + pre_label
    real_number = datas

    sim_tolerate_num = neighbers

    temp_number = torch.tensor(real_number[:, :-1], dtype=torch.float32)

    train_pred_prob = torch.tensor(train_pred_prob, dtype=torch.float32)

    import torch.nn.functional as F

    x = temp_number.unsqueeze(1)

    y = train_pred_prob.unsqueeze(0)

    chuck = 200

    data_lists = torch.chunk(x, chuck, dim=0)

    output = []
    for data in data_lists:
        output.append(F.cosine_similarity(data, y, dim=-1))

    sim_matrix = torch.cat(output,dim=0)

    #sim_matrix = F.cosine_similarity(x, y, dim=-1)

    print("The sim_matrix is shape {}".format(sim_matrix.shape))

    dir_sims = (1.0 - sim_matrix).numpy()

    no_equl_pairs, equal_pairs = get_catogery(real_number, dir_sims, sim_tolerate_num, y_train)

    suspincs_sample = set(no_equl_pairs)

    true_sampe = set(equal_pairs)

    print(f"total is {len(suspincs_sample)+len(true_sampe)}")

    true_sampe = set(filter(lambda x: x not in noise_idx and x not in suspincs_sample, true_sampe))
    return true_sampe

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_noises(ood_scores,threshold=0.999):

    noise_idx = np.where(ood_scores>= threshold)[0]
    return noise_idx


def slice(true_sampe, noise_idx, priorizations: List):
    '''
    :param true_sampe: 正标签的索引
    :param noise_idx: suspicious noise sample
    :param priorizations: [(uncertain, index, ood_score)]
    :return:
    '''
    # (uncertain, key)
    correct_array = []
    suspicious_array = []
    noise_array = []
    for priorization in priorizations:
        sample_index = priorization[1]
        if sample_index in true_sampe:
            correct_array.append(priorization)
        elif sample_index in noise_idx:
            noise_array.append(priorization)
        else:
            suspicious_array.append(priorization)

    assert len(noise_array) == len(noise_idx) and len(correct_array) == len(true_sampe)

    # according to  certain metric to obtain the priorization results.
    correct_array = sorted(correct_array, key=lambda x: x[0], reverse=True)
    noise_array = sorted(noise_array, key=lambda x: x[0], reverse=True)

    return suspicious_array + noise_array + correct_array


if __name__ == '__main__':
    pass
