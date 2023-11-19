# #!/usr/bin/env python2
# # -*- coding: utf-8 -*-
#
import time
from collections import defaultdict

import numpy as np
from keras import Model
from keras.backend import clear_session
from scipy import stats
from tqdm import tqdm


# 记录中间时间
# 如果 内存不够 改成批量预测
# use_batch_predict =True
class MetricsTime(object):
    def __init__(self, t_collection=0, t_cam=0, t_ctm=0):
        self.t_collection = t_collection  # 收集时间
        self.t_cam = t_cam  # cam排序时间
        self.t_ctm = t_ctm  # ctm排序时间
        self.first_add_info_time = True
        self.use_batch_predict = False  # 批量预测(内存不够)

    def get_t_collection(self):
        return self.t_collection

    def get_t_cam(self):
        return self.t_cam

    def get_t_ctm(self):
        return self.t_ctm

    def predict_by_batch(self, test, i):
        if self.use_batch_predict:
            len_test = len(test)
            arr = np.linspace(0, len_test, num=11)
            num_arr = [int(x) for x in arr]
            temp_arr = []
            for ix in range(len(num_arr) - 1):
                start_ix = num_arr[ix]
                end_ix = num_arr[ix + 1]
                temp = i.predict(test[start_ix:end_ix])
                temp.astype(np.float32)
                temp_arr.append(temp)
            temp_arr = np.concatenate(temp_arr, axis=0)
        else:
            temp_arr = i.predict(test)
        return temp_arr


## deep gauge
class kmnc(MetricsTime):
    def __init__(self, train, input, layers, k_bins=1000, max_select_size=None, time_limit=43200):
        super().__init__(t_ctm=None)
        s = time.time()
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train = train
        self.input = input
        self.layers = layers
        self.k_bins = k_bins
        self.lst = []
        self.upper = []
        self.lower = []
        index_lst = []

        # 资源限制
        self.time_limit = time_limit
        self.max_select_size = max_select_size

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.upper.append(np.max(temp, axis=0))  # 每层的最大值
            self.lower.append(np.min(temp, axis=0))  # 每层的最小值

        self.upper = np.concatenate(self.upper, axis=0)  # 将最大值按照行拼接到一起
        self.lower = np.concatenate(self.lower, axis=0)
        self.neuron_num = self.upper.shape[0]  # 每个神经元都有一个最大值
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self, test):
        '''
        test:测试集数据
        输出测试集的覆盖率
        '''
        self.neuron_activate = []  # 三维数组,里面的每个二维数组代表着一个temp
        # 一个temp是一层的输出值,高度代表每个测试用例,长度代表每个神经元
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)  # 3维变2维
        # 拼接之后高度代表每个测试用例,长度代表所有的神经元
        act_num = 0
        for index in range(len(self.upper)):
            bins = np.linspace(self.lower[index], self.upper[index], self.k_bins)  # 将每层的最大值和最小值按照bins,划分成多少个区间
            act_num += len(np.unique(np.digitize(self.neuron_activate[:, index], bins)))  # bug?
            # self.neuron_activate[:, index] 代表第i个神经元的所有测试用例的输出值,
            # np.digitize统计输出值所占据的区间并np.unique去重,这样就得到了第i个神经元所覆盖的区间
            # sum求和意味着覆盖的区间数
        return act_num / float(self.k_bins * self.neuron_num)

    # 获取测试集处在bins中的情况
    def get_big_bins(self, test):
        s = time.time()
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        big_bins = np.zeros((len(test), self.neuron_num, self.k_bins + 1))
        for n_index, neuron_activate in tqdm(enumerate(self.neuron_activate)):
            for index in range(len(neuron_activate)):
                bins = np.linspace(self.lower[index], self.upper[index], self.k_bins)
                temp = np.digitize(neuron_activate[index], bins)
                big_bins[n_index][index][temp] = 1

        big_bins = big_bins.astype('int')
        e = time.time()
        if self.first_add_info_time:
            self.t_collection += e - s
            self.first_add_info_time = False
        return big_bins

    def rank_fast(self, test):
        '''
        test:测试集数据
        输出排序情况
        '''
        big_bins = self.get_big_bins(test)  # 获取中间信息
        start = time.time()
        subset = []
        no_idx_arr = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))
        no_idx_arr.append(initial)
        subset.append(initial)
        max_cover_num = (big_bins[initial] > 0).sum()
        cover_last = big_bins[initial]
        while True:
            end = time.time()
            if self.max_select_size is not None and len(subset) == self.max_select_size:
                # 选够了,提前终止排序
                print("max_select_size", len(subset))
                self.t_cam = end - start
                return subset
            if end - start >= self.time_limit:
                # 时间到了,提前终止排序
                print("=======================time limit=======================")
                return subset
            print("剩余时间: {}".format(self.time_limit - (end - start)))
            flag = False
            for index in lst:
                if index in no_idx_arr:
                    continue
                temp1 = np.bitwise_or(cover_last, big_bins[index])
                now_cover_num = (temp1 > 0).sum()
                if now_cover_num > max_cover_num:
                    max_cover_num = now_cover_num
                    max_index = index
                    max_cover = temp1
                    flag = True
            cover_last = max_cover
            if not flag or len(lst) == 1:
                break
            no_idx_arr.append(max_index)
            subset.append(max_index)
            print(len(subset), end - start)
        self.t_cam = end - start
        return subset

    def rank_greedy(self, test):
        '''
              test:测试集数据
              输出排序情况
              '''
        big_bins = self.get_big_bins(test)  # 获取中间信息
        start = time.time()
        subset = []  # 返回结果
        lst = list(range(len(test)))  # 测试数据的下标顺序

        np.random.seed(0)

        initial = np.random.permutation(len(test))[0]  # 第一个作为初始队列里的数据
        subset.append(initial)

        max_cover_num = (big_bins[initial] > 0).sum()  # 计算当前队列的覆盖
        cover_last = big_bins[initial]  # 保存覆盖信息

        for index in lst:
            if index == initial:
                continue
            temp1 = np.bitwise_or(cover_last, big_bins[index])  # 计算当前的覆盖信息
            now_cover_num = (temp1 > 0).sum()  # 计算当前的覆盖率
            if now_cover_num > max_cover_num:
                max_cover_num = now_cover_num  # 更新最大覆盖率
                cover_last = temp1  # 更新最大覆盖信息
                subset.append(index)  # 添加最大覆盖标签
        end = time.time()
        self.t_cam = end - start

        return subset


class nbc(MetricsTime):
    def __init__(self, train, input, layers, std=0):
        super().__init__()
        s = time.time()
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train = train
        self.input = input
        self.layers = layers
        self.std = std
        self.lst = []
        self.upper = []
        self.lower = []
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = self.predict_by_batch(train, i).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(train, i, l.shape[-1])
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.upper.append(np.max(temp, axis=0) + std * np.std(temp, axis=0))
            self.lower.append(np.min(temp, axis=0) - std * np.std(temp, axis=0))
        self.upper = np.concatenate(self.upper, axis=0)
        self.lower = np.concatenate(self.lower, axis=0)
        self.neuron_num = self.upper.shape[0]
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self, test, use_lower=False):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, l, l.output.shape[-1])
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)

        act_num = 0
        act_num += (np.sum(self.neuron_activate > self.upper, axis=0) > 0).sum()
        if use_lower:
            act_num += (np.sum(self.neuron_activate < self.lower, axis=0) > 0).sum()

        if use_lower:
            return act_num / (2 * float(self.neuron_num))
        else:
            return act_num / float(self.neuron_num)

    # 获取测试集是否大于边界
    def get_lower_and_upper_flag(self, test, use_lower):
        s = time.time()
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, l, l.output.shape[-1])
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)

        upper = (self.neuron_activate > self.upper)
        if use_lower:
            lower = (self.neuron_activate < self.lower)
        else:
            lower = []

        e = time.time()
        if self.first_add_info_time:
            self.t_collection += e - s
            self.first_add_info_time = False
        return upper, lower

    def rank_fast(self, test, use_lower=False):
        upper, lower = self.get_lower_and_upper_flag(test, use_lower)  # 获取中间信息
        s = time.time()
        subset = []
        no_idx_arr = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))

        no_idx_arr.append(initial)
        subset.append(initial)
        max_cover_num = np.sum(upper[initial])
        if use_lower:
            max_cover_num += np.sum(lower[initial])
        cover_last_1 = upper[initial]
        if use_lower:
            cover_last_2 = lower[initial]
        while True:
            flag = False
            for index in lst:
                if index in no_idx_arr:  # 如果该数据已经考虑过,就不要再考虑了
                    continue
                temp1 = np.bitwise_or(cover_last_1, upper[index])
                cover1 = np.sum(temp1)
                if use_lower:
                    temp2 = np.bitwise_or(cover_last_2, lower[index])
                    cover1 += np.sum(temp2)
                if cover1 > max_cover_num:
                    max_cover_num = cover1
                    max_index = index
                    flag = True
                    max_cover1 = temp1
                    if use_lower:
                        max_cover2 = temp2
            if not flag or len(lst) == 1:
                break
            no_idx_arr.append(max_index)
            # lst.remove(max_index)
            subset.append(max_index)
            cover_last_1 = max_cover1
            if use_lower:
                cover_last_2 = max_cover2
            # print(max_cover_num)
        e = time.time()
        self.t_cam = e - s
        return subset

    def rank_2(self, test, use_lower=False):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, l, l.output.shape[-1])
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        if use_lower:
            s = time.time()
            res = np.argsort(
                np.sum(self.neuron_activate > self.upper, axis=1) + np.sum(self.neuron_activate < self.lower, axis=1))[
                  ::-1]
            e = time.time()
        else:
            s = time.time()
            res = np.argsort(np.sum(self.neuron_activate > self.upper, axis=1))[::-1]
            e = time.time()
        self.t_ctm = e - s
        return res


class tknc(MetricsTime):
    def __init__(self, test, input, layers, k=2):
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = test
        self.input = input
        self.layers = layers
        self.k = k
        self.lst = []
        self.neuron_activate = []
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = self.predict_by_batch(test, i).reshape(len(test), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, i, l.shape[-1])
            if index == 'dense':
                temp = i.predict(test).reshape(len(test), l.shape[-1])
            self.neuron_activate.append(temp)
        self.neuron_num = np.concatenate(self.neuron_activate, axis=1).shape[-1]
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self, choice_index):
        neuron_activate = 0
        for neu in self.neuron_activate:
            temp = neu[choice_index]
            neuron_activate += len(np.unique(np.argsort(temp, axis=1)[:, -self.k:]))
        return neuron_activate / float(self.neuron_num)

    # 获得测试集里前k个最大覆盖的神经元
    def get_top_k_neuron(self):
        s = time.time()
        neuron = []
        layers_num = 0
        for neu in self.neuron_activate:
            neuron.append(np.argsort(neu, axis=1)[:, -self.k:] + layers_num)
            layers_num += neu.shape[-1]
        neuron = np.concatenate(neuron, axis=1)
        e = time.time()
        self.t_collection += e - s
        return neuron

    def rank(self, test):
        neuron = self.get_top_k_neuron()  # 获得信息

        s = time.time()
        subset = []
        no_idx_arr = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))

        no_idx_arr.append(initial)
        # lst.remove(initial)
        subset.append(initial)
        max_cover = len(np.unique(neuron[initial]))

        cover_now = neuron[initial]

        while True:
            flag = False
            for index in lst:
                if index in no_idx_arr:  # 如果该数据已经考虑过,就不要再考虑了
                    continue
                temp = np.union1d(cover_now, neuron[index])
                cover1 = len(temp)
                if cover1 > max_cover:
                    max_cover = cover1
                    max_index = index
                    flag = True
                    max_cover_now = temp
            if not flag or len(lst) == 1:
                break
            # lst.remove(max_index)
            no_idx_arr.append(max_index)
            subset.append(max_index)
            cover_now = max_cover_now
            # print(max_cover)
        e = time.time()
        self.t_cam = e - s
        return subset


## deepxplore
class nac(MetricsTime):
    def __init__(self, test, input, layers, t=0):
        super().__init__()
        s = time.time()
        self.train = test
        self.input = input
        self.layers = layers
        self.t = t
        self.lst = []
        self.neuron_activate = []
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            i.predict(np.zeros((1, *self.train.shape[1:])))
            if index == 'conv':
                # temp = i.predict(test).reshape(len(test), -1, l.shape[-1])
                temp = self.predict_by_batch(test, i).reshape(len(test), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, i, l.shape[-1])
            if index == 'dense':
                temp = i.predict(test).reshape(len(test), l.shape[-1])
            temp = 1 / (1 + np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_num = np.concatenate(self.neuron_activate, axis=1).shape[-1]
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self):
        neuron_activate = 0
        for neu in self.neuron_activate:
            neuron_activate += np.sum(np.sum(neu > self.t, axis=0) > 0)
        return neuron_activate / float(self.neuron_num)

    # 获取测试集中的神经元激活情况
    def get_upper(self, test):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, l, l.output.shape[-1])
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            temp = 1 / (1 + np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        s = time.time()
        upper = (self.neuron_activate > self.t)  # upper是一个TrueFalse 矩阵,shape(用例数,神经元数)
        e = time.time()
        self.t_collection += e - s
        return upper

    def rank_fast(self, test):
        upper = self.get_upper(test)  # 获取中间信息
        s = time.time()
        # 注释:类似于贪心算法
        # 思想,先随机选一个,记录激活的神经元,在顺序遍历所有的例子,找到一个和原有例子放在一起后
        # 可以激活最多个数神经元的例子,找到后就放在数组里记录它的索引
        # 直到没有神经元找得到或者全找完了为止

        subset = []  # 存放最终的选择序列
        no_idx_arr = []
        lst = list(range(len(test)))  # 存放所有的测试集的索引
        initial = np.random.choice(range(len(test)))  ## initial是随机选择了一个测试用例的下标

        # lst.remove(initial)
        no_idx_arr.append(initial)
        subset.append(initial)  # 将下标从lst移除,添加到subset中  下次就不会遍历到它
        max_cover_num = np.sum(upper[initial])  # 统计激活了的神经元的个数
        cover_last_1 = upper[initial]  # 选取了第x个测试用例对应的全部神经元的激活状态

        # 遍历剩下的用例
        while True:
            flag = False
            for index in lst:
                if index in no_idx_arr:
                    continue
                # cover_last_1 定义过,就是第x个的用例的神经元激活状态,
                # upper[index] 是第i个用例的神经元激活状态
                # 两者求异或,意味着两者中对应神经元位置有一个激活则算作激活
                temp1 = np.bitwise_or(cover_last_1, upper[index])
                cover1 = np.sum(temp1)  # 求激活了的神经元数量
                if cover1 > max_cover_num:  # 如果激活了的数量大于之前的神经元激活数量
                    max_cover_num = cover1  # 将最大激活数量改为新值
                    max_index = index  # 记录下来当前例子的索引
                    flag = True  # 继续循环
                    max_cover1 = temp1  # 添加了新用例之后的神经元激活状态
            # 至此结束,每次选择了一个新的用例,保证此时激活了的神经元最多,并添加进数组
            if not flag or len(lst) == 1:
                break
            # 如果找到了新的用例,或者还有1个用例以上,就继续循环  #找不到用例意味着满足了最大覆盖,到1个用例意味着全选完了
            no_idx_arr.append(max_index)
            subset.append(max_index)  # 添加进来
            cover_last_1 = max_cover1  # 赋值,
            # print(max_cover_num)
        e = time.time()
        self.t_cam = e - s
        return subset

    def rank_2(self, test):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, l, l.output.shape[-1])
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            temp = 1 / (1 + np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)

        s = time.time()
        res = np.argsort(np.sum(self.neuron_activate > self.t, axis=1))[::-1]
        e = time.time()
        self.t_ctm = e - s
        return res


## deepgini
def deep_metric(pred_test_prob):
    metrics = np.sum(pred_test_prob ** 2, axis=1)  # 值越小,1-值就越大,因此值越小越好
    rank_lst = np.argsort(metrics)  # 按照值从小到大排序,因此序号越小代表值越小代表越好
    return rank_lst


def deep_metric2(pred_test_prob):
    metrics = np.sum(pred_test_prob ** 2, axis=1)  # 值越小,1-值就越大,因此值越小越好
    rank_lst = np.argsort(metrics)  # 按照值从小到大排序,因此序号越小代表值越小代表越好
    return rank_lst, 1 - metrics


# Surprise Adequacy
# LSC
# 这里的u一定要手动设置一个合适的2000/100/150/5
# LSC不能使用所有的层来计算,计算量太大了,因此在使用前必须选好层
# 作者只选了某一层


class LSC(MetricsTime):
    def __init__(self, train, label, input, layers, u=2000, k_bins=1000, threshold=None):

        """
        :param train:  训练集数据
        :param label:
        :param input:  输入张量
        :param layers:  输出张量层
        :param u:  上界
        :param k_bins:  分割段数
        :param threshold:  阈值筛选
        """
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = train
        self.input = input
        self.layers = layers
        self.lst = []
        self.neuron_activate_train = []
        self.u = u
        self.k_bins = k_bins
        self.threshold = threshold
        self.test_score = []
        self.train_label = np.array(label)
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            temp = None
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])  # len(train), h*w num(filters)
                temp = np.mean(temp, axis=1)  # 对H*w里也就是一个filter里的所有神经元就平均值 len(train), l.shape[-1]
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.neuron_activate_train.append(temp.copy())  # len(train), l.shape[-1]
        self.neuron_activate_train = np.concatenate(self.neuron_activate_train, axis=1)  #
        self.lst = list(zip(index_lst, self.lst))
        # print("LSC init over")
        e = time.time()
        self.t_collection += e - s

        self.class_matrix = None
        self._init_class_matrix()
        self.kdes, self.removed_cols = self._init_kdes()

    def _init_class_matrix(self):
        # 按照lable分类
        class_matrix = {}
        for i, lb in enumerate(self.train_label):
            if lb not in class_matrix:
                class_matrix[lb] = []
            class_matrix[lb].append(i)
        self.class_matrix = class_matrix

    def _init_kdes(self):
        class_matrix = self.class_matrix
        train_ats = self.neuron_activate_train
        num_classes = np.unique(self.train_label)
        removed_cols = []
        if self.threshold is not None:
            for lb in num_classes:
                col_vectors = np.transpose(train_ats[class_matrix[lb]])   # 转置
                for i in range(col_vectors.shape[0]):
                    if (
                            np.var(col_vectors[i]) < self.threshold  # 方差过滤
                            and i not in removed_cols
                    ):
                        removed_cols.append(i)

        kdes = {}
        for lb in num_classes:
            refined_ats = np.transpose(train_ats[class_matrix[lb]])  # 只选出符合lable的激活迹  # 这里注意在构建kde时,有转置
            refined_ats = np.delete(refined_ats, removed_cols, axis=0)  # 删除掉不符合方差的
            # 删除后shape(过滤后神经元,测试用例)
            if refined_ats.shape[0] == 0:  # 全都移除了
                print("warning....  remove all")
                continue
            kdes[lb] = stats.gaussian_kde(refined_ats)
        return kdes, removed_cols

    def _get_lsa(self, kde, at, removed_cols):
        refined_at = np.delete(at, removed_cols, axis=0)  # 对与测试集, 删除掉那些方差不符合规则的激活迹.
        # refined_at 是测试集中某个测试用例删除掉不符合规则的激活迹
        return np.asscalar(-kde.logpdf(np.transpose(refined_at)))  # 计算激活迹的-log值 即LSA值,这里注意代码中有转置

    # 1. fit 获取覆盖率
    # 2. 获取中间信息
    def fit(self, test, label):
        s = time.time()
        # print("LSC fit")
        self.neuron_activate_test = []
        self.test_score = []
        for index, l in self.lst:
            temp = None
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())
        self.neuron_activate_test = np.concatenate(self.neuron_activate_test, axis=1)  # 10000 10

        # 按照lable分类
        class_matrix = self._init_class_matrix()

        kdes, removed_cols = self.kdes, self.removed_cols  # 获得了训练集的激活迹

        for test_sample, label_sample in tqdm(zip(self.neuron_activate_test, label)):  # 对于每一个测试集样本
            if label_sample in kdes.keys():
                kde = kdes[label_sample]  # 选出该标签对应的训练集的激活迹
                self.test_score.append(self._get_lsa(kde, test_sample, removed_cols))  # 添加到lsa数组中
            else:
                self.test_score.append(0)
        e = time.time()
        self.t_collection += e - s
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        x = np.unique(np.digitize(self.test_score, bins))
        rate = len(np.unique(x)) / float(self.k_bins)
        return rate

    def get_sore(self):
        return self.test_score

    def get_u(self):
        return self.u

    def rank_fast(self):
        s = time.time()
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        score_bin = np.digitize(self.test_score, bins)
        score_bin_uni = np.unique(score_bin)
        res_idx_arr = []
        for x in score_bin_uni:
            np.random.seed(41)
            idx_arr = np.argwhere(score_bin == x).flatten()
            idx = np.random.choice(idx_arr)
            res_idx_arr.append(idx)
        # print(len(res_idx_arr), self.k_bins * self.get_rate())
        e = time.time()
        self.t_cam = e - s
        return res_idx_arr

    def rank_2(self):
        s = time.time()
        res = np.argsort(self.get_sore())[::-1]  # 由大到小排序
        e = time.time()
        self.t_ctm = e - s
        return res


## DSC
class DSC(MetricsTime):
    def __init__(self, train, label, input, layers, u=2, k_bins=1000, threshold=10 ** -5, time_limit=3600):
        '''
        train:训练集数据
        label:训练集的标签
        input:输入张量
        layers:输出张量层
        std : 方差筛选
        u : 上界
        k_bins: 分割段数
        threshold: 阈值筛选
        '''
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = train
        self.input = input
        self.layers = layers
        self.lst = []
        self.std_lst = []
        self.mask = []
        self.neuron_activate_train = []
        index_lst = []
        self.u = u
        self.k_bins = k_bins
        self.threshold = threshold
        self.test_score = []

        self.time_limit = time_limit

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.neuron_activate_train.append(temp.copy())
        self.neuron_activate_train = np.concatenate(self.neuron_activate_train, axis=1)
        self.train_label = np.array(label)
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def find_closest_at(self, at, train_ats):
        dist = np.linalg.norm(at - train_ats, axis=1)  # 二范数值
        return (min(dist), train_ats[np.argmin(dist)])  # 找到二范数值最近的,同时把结果返回去

    def fit(self, test, label):
        s = time.time()
        start = time.time()
        self.neuron_activate_test = []
        self.test_score = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())
        self.neuron_activate_test = np.concatenate(self.neuron_activate_test, axis=1)

        class_matrix = {}
        all_idx = []
        for i, lb in enumerate(self.train_label):
            if lb not in class_matrix:
                class_matrix[lb] = []
            class_matrix[lb].append(i)
            all_idx.append(i)

        # dsa代码
        for test_sample, label_sample in tqdm(zip(self.neuron_activate_test, label)):
            end = time.time()
            if end - start >= self.time_limit:
                print("=======================time limit=======================")
                return None
            x = self.neuron_activate_train[class_matrix[label_sample]]
            a_dist, a_dot = self.find_closest_at(test_sample, x)
            y = self.neuron_activate_train[list(set(all_idx) - set(class_matrix[label_sample]))]
            b_dist, _ = self.find_closest_at(
                a_dot, y
            )  # 求出最近的距离值
            self.test_score.append(a_dist / b_dist)

        e = time.time()
        self.t_collection += e - s
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        x = np.unique(np.digitize(self.test_score, bins))
        rate = len(np.unique(x)) / float(self.k_bins)
        return rate

    def get_sore(self):
        return self.test_score

    def get_u(self):
        return self.u

    def rank_2(self):
        s = time.time()
        res = np.argsort(self.get_sore())[::-1]  # 由大到小排序
        e = time.time()
        self.t_ctm = e - s
        return res

    def rank_fast(self):
        s = time.time()
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        score_bin = np.digitize(self.test_score, bins)
        score_bin_uni = np.unique(score_bin)
        res_idx_arr = []
        for x in score_bin_uni:
            np.random.seed(41)
            idx_arr = np.argwhere(score_bin == x).flatten()
            idx = np.random.choice(idx_arr)
            res_idx_arr.append(idx)
        e = time.time()
        self.t_cam = e - s
        return res_idx_arr


class MCDC(object):
    def __init__(self, train, input, layers, d_vc=None, d_dc=None):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train = train
        self.input = input
        self.layers = layers
        self.lst = []
        self.neuron_activate_train = []
        index_lst = []
        # mine......
        self.n = len(layers)  # 层数
        self.neuron_num_arr = []  # 每层神经元的个数
        self.neuron_pair_num = 0  # 神经元对的个数
        self.case_dict = defaultdict(set)  # 存放用例与神经元的关系
        self.neuron_con_arr = [0]  # 神经元的连接数,第一层的连接数是0,因此写0
        self.neuron_map_arr = []  # 神经元具体的连接情况
        self.neuron_activate_test = []  # shape(层数,)  里面的list, shape(测试集个数,神经元个数)
        # 值变化,暂时不用
        self.d_vc = d_vc
        self.d_dc = d_dc
        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            temp = None
            if index == 'conv':
                # 修改一下
                # temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                # temp = np.mean(temp, axis=1)
                temp = i.predict(train).reshape(len(train), -1)
            if index == 'dense':
                # print(i.predict(train).shape)
                # print(l.shape)
                temp = i.predict(train).reshape(len(train), l.shape[-1])

            self.neuron_activate_train.append(temp.copy())
            self.neuron_num_arr.append(temp.shape[-1])
        self.neuron_activate_train = np.concatenate(self.neuron_activate_train, axis=1)
        self.lst = list(zip(index_lst, self.lst))
        for i in range(len(self.neuron_num_arr) - 1):
            temp = self.neuron_num_arr[i] * self.neuron_num_arr[i + 1]  # 每相邻两层的神经元相乘,计算出连接数,也就是神经元对数
            self.neuron_pair_num += temp
            self.neuron_con_arr.append(temp)
            self.neuron_map_arr.append(np.zeros((self.neuron_num_arr[i], self.neuron_num_arr[i + 1])))

    # 获取值对应的符号
    def _get_sign(self, test):
        return np.where(test > 0, 1, 0)  # 大于0的设置为1 # 小于等于0的设置为0

    # 用于去重,
    def _get_pre_and_cur(self, pre_matrix, cur_matrix):
        '''
        :param pre_matrix:   去重前矩阵1
        :param cur_matrix:  去重前矩阵2
        :return:  去重的行1,去重的行2,去重后矩阵1,去重后矩阵2
        '''
        pre_matrix_row = pre_matrix[0]
        cur_matrix_row = cur_matrix[0]
        pre_matrix_now = np.delete(pre_matrix_row, 0, axis=0)  # 每次删除最上面一行
        cur_matrix_now = np.delete(cur_matrix_row, 0, axis=0)  # 每次删除最上面一行
        return pre_matrix_row, cur_matrix_row, pre_matrix_now, cur_matrix_now

    # 用于储存用例与用例之间激活的神经元和整个神经元的激活情况
    def _save_map_and_case(self, pre_neu_indx_arr, cur_neu_indx_arr, case_index, layer_num, cur_case_index):
        '''
        :param pre_neu_indx_arr:  上一层神经元所在位置
        :param cur_neu_indx_arr:   本层神经元所在位置
        :param case_index:   与第x个用例满足变化的用例
        :param layer_num:  现在的层数
        :param cur_case_inde: 第x个用例
        :return:
        '''
        for ix, pre_neu_indx in enumerate(pre_neu_indx_arr):
            cur_neu_indx = cur_neu_indx_arr[cur_neu_indx_arr[:, 0] == ix][:, 1]  # 当前神经元的下标
            if len(cur_neu_indx) != 0:
                self.neuron_map_arr[layer_num - 1][pre_neu_indx, cur_neu_indx] = 1  #
                # print(("第{}/{}层: 用例({},{})--->在神经元({}与{})上满足ssc覆盖".format(i - 1, i, j, case_index[ix],
                #                                                           pre_neu_indx,
                #                                                           cur_neu_indx)))
                for x in cur_neu_indx:
                    key = None
                    if cur_case_index < case_index[ix]:
                        key = str(cur_case_index) + "_" + str(case_index[ix])
                    else:
                        key = str(case_index[ix]) + "_" + str(cur_case_index)
                    self.case_dict[key].add((layer_num - 1, pre_neu_indx, x))
                    # 连接数 = 前层连接数+ 同层前面的神经元个数 * 下一层所有神经元的个数 + 下一层当前神经元的个数
                    # self.case_dic

    # 获取符号差, 求出符号变化矩阵  只要求出不一样的值就行
    def _get_sc_dif_matrix(self, matrix, row):
        return np.abs(matrix - row)  # 符号差

    # 根据上一层中发生变化的用例的下标
    # 寻找出选出本层中的对应用例
    def _get_neu_indx_arr(self, pre_sign_temp, cur_sign_temp, case_index):
        '''
        :param pre_sign_temp:  上一层的符号差
        :param cur_sign_temp:   本层的符号差
        :param  case_index 用例所在的下标
        :return:     两层神经元所在的位置
        '''
        # 选取那些行中,值为1的列,记录其索引,这里只记录第二行即可,因为个用例在每行只能有一个神经元被激活
        pre_neu_indx_arr = (np.argwhere(pre_sign_temp[case_index])[:, 1])

        # 横坐标代表第x个用例,纵坐标代表第y个神经元
        # 这里要全部保留,因为每个用例在每行可能有多个神经元被激活
        cur_neu_indx_arr = (np.argwhere(cur_sign_temp[case_index]))  # 元组的集合 元组(第x个用例,第y个神经元)
        return pre_neu_indx_arr, cur_neu_indx_arr

    def ssc(self):
        #################################################################################################
        # 1. 这次我们要在遍历值数组的同时,统计出神经元对与用例的关系
        # 2. 每次要遍历两层,为了避免重复计算,先计算出第一层,从第二层再开始遍历
        pre_test = None
        pre_sign = None
        for i in tqdm(range(len(self.neuron_activate_test))):  # 遍历值数组
            if i == 0:
                pre_test = self.neuron_activate_test[0]
                pre_sign = self._get_sign(pre_test)
                continue  # 从第二层开始算
            cur_test = self.neuron_activate_test[i]
            cur_sign = self._get_sign(cur_test)

            pre_sign_now = pre_sign.copy()  # 备份副本,用于去重
            cur_sign_now = cur_sign.copy()

            #  构造sc值矩阵
            for j in tqdm(range(len(pre_test) - 1)):  # 修改,增加去重
                # 去重预处理
                pre_sign_row, cur_sign_row, pre_sign_now, cur_sign_now = self._get_pre_and_cur(pre_sign_now,
                                                                                               cur_sign_now)
                # 求出前一层符号差
                pre_sign_temp = self._get_sc_dif_matrix(pre_sign_now, pre_sign_row)

                # 根据前一层的符号差计算出前一层所有用例与第j个用例在所有神经元上的变化数量,
                # 某个值为1代表发生了变化,每行为1的值求和代表前一层某些用例与第j个用例在神经元上发生的变化数量
                # 和为1代表只有一次变化,:即符合mc条件
                pre_sc_num = np.sum(pre_sign_temp == 1, axis=1)

                # 将和为1对应的用例索引取出
                case_index = np.argwhere(pre_sc_num == 1).flatten()  # 选出上一层符号变化为1的用例所对应的索引

                # 求出本层符号差
                cur_sign_temp = self._get_sc_dif_matrix(cur_sign_now, cur_sign_row)  # 本层的符号差

                # 根据两层符号差,和对应的用例索引,取出变化了的神经元下标
                pre_neu_indx_arr, cur_neu_indx_arr = self._get_neu_indx_arr(pre_sign_temp, cur_sign_temp, case_index)

                # 根据神经元下标将值写入内存
                # 计算第几个用例的时候,要把删除的行数加回去
                self._save_map_and_case(pre_neu_indx_arr, cur_neu_indx_arr, case_index + j + 1, i - 1, j)
            pre_test = cur_test
            pre_sign = cur_sign

        sum = 0
        for x in self.neuron_map_arr:
            l_sum = np.sum(x == 1)
            sum += l_sum
        print("神经元激活情况")
        print(self.neuron_map_arr)
        # print("用例之间激活的神经元个数")
        # print(self.case_dict)

        return sum / self.neuron_pair_num

    # modify
    def svc(self, d_vc=None):
        #################################################################################################
        # 1. 这次我们要在遍历值数组的同时,统计出神经元对与用例的关系
        # 2. 每次要遍历两层,为了避免重复计算,先计算出第一层,从第二层再开始遍历
        pre_test = None
        pre_sign = None
        for i in tqdm(range(len(self.neuron_activate_test))):  # 遍历值数组
            if i == 0:
                pre_test = self.neuron_activate_test[0]
                pre_sign = self._get_sign(pre_test)
                continue  # 从第二层开始算
            cur_test = self.neuron_activate_test[i]
            cur_sign = self._get_sign(cur_test)

            pre_sign_now = pre_sign.copy()  # 备份副本,用于去重
            cur_sign_now = cur_sign.copy()
            #  构造sc值矩阵
            for j in tqdm(range(len(pre_test) - 1)):  # 修改,增加去重
                # 去重预处理
                pre_sign_row, cur_sign_row, pre_sign_now, cur_sign_now = self._get_pre_and_cur(pre_sign_now,
                                                                                               cur_sign_now)
                # 求出前一层符号差
                pre_sign_temp = self._get_sc_dif_matrix(pre_sign_now, pre_sign_row)

                # 根据前一层的符号差计算出前一层所有用例与第j个用例在所有神经元上的变化数量,
                # 某个值为1代表发生了变化,每行为1的值求和代表前一层某些用例与第j个用例在神经元上发生的变化数量
                # 和为1代表只有一次变化,:即符合mc条件
                pre_sc_num = np.sum(pre_sign_temp == 1, axis=1)

                # 将和为1对应的用例索引取出
                case_index = np.argwhere(pre_sc_num == 1).flatten()  # 选出上一层符号变化为1的用例所对应的索引

                # 求出本层符号差
                cur_sign_temp = self._get_sc_dif_matrix(cur_sign_now, cur_sign_row)  # 本层的符号差

                # 根据两层符号差,和对应的用例索引,取出变化了的神经元下标
                pre_neu_indx_arr, cur_neu_indx_arr = self._get_neu_indx_arr(pre_sign_temp, cur_sign_temp, case_index)

                # 根据神经元下标将值写入内存
                # 计算第几个用例的时候,要把删除的行数加回去
                self._save_map_and_case(pre_neu_indx_arr, cur_neu_indx_arr, case_index + j + 1, i - 1, j)
            pre_test = cur_test
            pre_sign = cur_sign

        sum = 0
        for x in self.neuron_map_arr:
            l_sum = np.sum(x == 1)
            sum += l_sum
        print("神经元激活情况")
        print(self.neuron_map_arr)
        # print("用例之间激活的神经元个数")
        # print(self.case_dict)

    def fit(self, test):
        for index, l in self.lst:
            temp = None
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())  # 构造值数组
        #################################################################################################
        # 1. 这次我们要在遍历值数组的同时,统计出神经元对与用例的关系
        # 2. 每次要遍历两层,为了避免重复计算,先计算出第一层,从第二层再开始遍历
        pre_test = None
        pre_sign = None
        for i in tqdm(range(len(self.neuron_activate_test))):  # 遍历值数组
            if i == 0:
                pre_test = self.neuron_activate_test[0]  #
                pre_sign = np.where(pre_test > 0, 1, 0)  # 大于0的设置为1 # 小于等于0的设置为0
                continue  # 从第二层开始算
            cur_test = self.neuron_activate_test[i]
            cur_sign = np.where(cur_test > 0, 1, 0)

            pre_sign_now = pre_sign.copy()  # 备份副本,用于去重
            cur_sign_now = cur_sign.copy()

            #  构造sc值矩阵
            for j in tqdm(range(len(pre_test) - 1)):  # 修改,增加去重
                # 去重预处理
                pre_sign_row = pre_sign_now[0]  # 为了避免重复,再比较完第一个用例后,将第一个用例删除掉
                cur_sign_row = cur_sign_now[0]
                pre_sign_now = np.delete(pre_sign_now, 0, axis=0)  # 每次删除最上面一行
                cur_sign_now = np.delete(cur_sign_now, 0, axis=0)  # 每次删除最上面一行

                pre_sign_temp = np.abs(pre_sign_now - pre_sign_row)  # 上一层的符号差
                pre_sc_num = np.sum(pre_sign_temp == 1, axis=1)  # 按行求出为1的个数,这个时候就知道了上一层那些用例符合符号变化

                cur_sign_temp = np.abs(cur_sign_now - cur_sign_row)
                case_index = np.argwhere(pre_sc_num == 1).flatten()  # 选出上一层符号变化为1的用例所对应的索引

                # 选取那些行中,值为1的列,记录其索引
                pre_neu_indx_arr = (np.argwhere(pre_sign_temp[case_index])[:, 1])
                # 横坐标代表第x个用例,纵坐标代表第y个神经元

                # 根据上一层中发生变化的用例的下标
                # 寻找出选出本层中的对应用例
                cur_neu_indx_arr = (np.argwhere(cur_sign_temp[case_index]))  # 元组的集合 元组(第x个用例,第y个神经元)

                case_index = case_index + j + 1  # 计算第几个用例的时候,要把删除的行数加回去
                for ix, pre_neu_indx in enumerate(pre_neu_indx_arr):
                    cur_neu_indx = cur_neu_indx_arr[cur_neu_indx_arr[:, 0] == ix][:, 1]  # 当前神经元的下标
                    if len(cur_neu_indx) != 0:
                        self.neuron_map_arr[i - 1][pre_neu_indx, cur_neu_indx] = 1  #
                        # 注释掉打印信息,
                        # print(("第{}/{}层: 用例({},{})--->在神经元({}与{})上满足ssc覆盖".format(i - 1, i, j, case_index[ix],
                        #                                                           pre_neu_indx,
                        #                                                           cur_neu_indx)))
                        # TD: 注释掉用例关系
                        # for x in cur_neu_indx:
                        #     key = None
                        #     if j < case_index[ix]:
                        #         key = str(j) + "_" + str(case_index[ix])
                        #     else:
                        #         key = str(case_index[ix]) + "_" + str(j)
                        #     self.case_dict[key].add((i - 1, pre_neu_indx, x))
                        # 连接数 = 前层连接数+ 同层前面的神经元个数 * 下一层所有神经元的个数 + 下一层当前神经元的个数
                        # self.case_dict[key].add(pre_neu_indx * self.neuron_con_arr[i - 1] + case_index[ix])
            print("第{}/{}层:神经元激活情况".format(i - 1, i))
            print(self.neuron_map_arr[i - 1])
            print("===========================")
            pre_test = cur_test
            pre_sign = cur_sign

        sum = 0
        for x in self.neuron_map_arr:
            l_sum = np.sum(x == 1)
            print(l_sum / (x.shape[0] * x.shape[1]))
            sum += l_sum
        # print("神经元激活情况") # 注释掉Map信息
        # print(self.neuron_map_arr)
        # print("用例之间激活的神经元个数")
        # print(self.case_dict)

        return sum / self.neuron_pair_num

    # 这个是之前的,已经废弃.原有的方法不能取除掉重复的神经元对,也不能用于排序1,只能废弃掉
    def fit2(self, test):
        self.neuron_activate_test = []  # shape(层数,)  里面的list, shape(测试集个数,神经元个数)
        for index, l in self.lst:
            temp = None
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())  # 构造值数组
        # shape(层数, ) 里面的list shape(测试集个数,测试集个数) 每个格子代表两个测试集在某一层发生变化的个数or是否发生变化
        self.sc_dif_array = []  # 符号变化的个数
        self.vc_dif_array = []  # 值变化的个数
        self.dc_dif_array = []  # 是否发生距离变化
        for i in range(len(self.neuron_activate_test)):  # 遍历值数组
            cur_test = self.neuron_activate_test[i]  # 当前层的测试集 shape(测试集个数,神经元个数)
            # 创建三个数组,用来存放三种变化
            sc_arr = []  # (测试集个数, )   横坐标代表选中了第x个测试集,纵坐标代表第y个测试集与第x个测试集的差异
            vc_arr = []
            dc_arr = []
            #  构造sc值矩阵
            cur_sign = np.where(cur_test > 0, 1, 0)  # 大于0的设置为1 # 小于等于0的设置为0
            for j, cur_case in enumerate(cur_test):  # 遍历每层的神经元的测试集 cur_case shape(神经元个数, )
                # sc
                cur_sign_temp = np.abs(cur_sign - cur_sign[j])  # 求差,这样0-0 =0  1-1=0 0-1=1 1-0=1 1代表变化 0代表没变化
                sc_num = np.sum(cur_sign_temp == 1, axis=1)  # 按行求出为1的个数 # shape(测试集个数,)
                sc_arr.append(sc_num)
                cur_sign_vc_mask = None
                cur_vc_temp = None
                if self.d_vc is not None:
                    # vc  使用"绝对"值变化,要求差值大于阈值,但是符号不变
                    if cur_vc_temp is None:  # 判断下,因为dc要复用这个变量
                        cur_vc_temp = np.fabs(cur_test - cur_test[j])  # 求当前测试集差
                    # 1.过滤掉没有符号变化的数
                    if cur_sign_vc_mask is None:  # 判断下,因为dc要复用这个变量
                        cur_sign_vc_mask = np.where(cur_sign_temp > 0, 0, 1)  # 对sc变化矩阵求反 1->0 0->1,即0代表变化 1代表没变化
                    cur_vc_sc_changed = cur_sign_vc_mask * cur_vc_temp  # 乘以距离sc变化矩阵,发生过变化的会变成0,这样有数值的一定是没有发生过sc变化的
                    # 2.求出差值大于某个距离的个数,记作符号变化的个数
                    vc_sum = np.sum(cur_vc_sc_changed >= self.d_vc, axis=1)
                    vc_arr.append(vc_sum)
                if self.d_dc is not None:
                    # dc  要求某层距离函数变化,但某层的符号全不能变化
                    # 1.某层符号不变化,意味着cur_sign_vc_mask矩阵某一行全是1, 1代表没变化,即全都没变化
                    # 按行求最小值,最小值为1则没变化 得到一个列向量,1代表该行没变化, 0代表该行变化了
                    if cur_sign_vc_mask is None:
                        cur_sign_vc_mask = np.where(cur_sign_temp > 0, 0, 1)
                    cur_sign_sc_mask = np.min(cur_sign_vc_mask, axis=1, keepdims=True)
                    if cur_vc_temp is None:
                        mr_dc_temp = np.fabs(cur_test - cur_test[j])
                    else:
                        mr_dc_temp = cur_vc_temp.copy()
                    mr_dc_temp = mr_dc_temp * cur_sign_sc_mask  # 相乘则过滤掉了变化的行
                    x_norm = np.linalg.norm(mr_dc_temp, ord=np.inf, axis=1, keepdims=False)  # 每行求范数 这里使用l无穷范数 就是最大值
                    dc_num = np.where(x_norm > self.d_dc, 1, 0)  # 大于阈值的设置1,小于的设置为0,  shape(测试集个数,)
                    dc_arr.append(dc_num)

            self.sc_dif_array.append(sc_arr)
            self.vc_dif_array.append(vc_arr)
            self.dc_dif_array.append(dc_arr)

        self.sc_dif_array = np.triu(np.array(self.sc_dif_array))  # 转化为numpy数组,因为数组是对称的,这里只要上三角
        self.dc_dif_array = np.triu(np.array(self.dc_dif_array))
        self.vc_dif_array = np.triu(np.array(self.vc_dif_array))

    def rank2_ssc(self):
        # return np.argsort(self.ssc_arr)[::-1]
        pass

    # def ssc(self, ):
    #     ssc_count = 0
    #     for i in range(self.n - 1):  # 遍历前n-1层,因为最后一层无法与下一层组成神经元对了
    #         # 取出前后两层
    #         sc_cur = self.sc_dif_array[i]  # shape (测试集个数,测试集个数)  某个格子等于1代表两个用例之间只有一个符号变化
    #         sc_next = self.sc_dif_array[i + 1]
    #         ssc_num = (sc_next[sc_cur == 1])  # 激活了的神经元对数量
    #         np.append(self.ssc_arr, ssc_num)  # 将神经元对数量添加到数组中,用于最后的排序
    #         ssc_count += np.sum(ssc_num)  # 上一层某两个用例发生了符号变化,统计下一层对应两个用例发生的符号变化数量的总和
    #     return ssc_count / self.neuron_pair_num

    def svc(self, ):
        svc_count = 0
        for i in range(self.n - 1):  # 遍历前n-1层
            sc_cur = self.sc_dif_array[i]  # 第一层取sc
            vc_next = self.vc_dif_array[i + 1]  # 第二层取vc
            svc_num = (vc_next[sc_cur == 1])
            np.append(self.svc_arr, svc_num)
            svc_count += np.sum(svc_num)  # 第一层满足sc的用例,统计在第二层中满足vc的个数
        return svc_count / self.neuron_pair_num

    def dsc(self):
        dsc_count = 0
        for i in range(self.n - 1):  # 遍历前n-1层
            dc_cur = self.dc_dif_array[i]  # 第一层取dc
            sc_next = self.sc_dif_array[i + 1]  # 第二层取sc
            dsc_num = (sc_next[dc_cur == 1])
            np.append(self.dsc_arr, dsc_num)
            dsc_count_temp = np.sum(dsc_num)  # 第一层满足dc的用例,统计在第二层中满足sc的个数
            dsc_count += (dsc_count_temp * self.neuron_num_arr[i])  # 根据定义: 第二层发生变化的个数 * 当前层的元素个数 = 总共发生变化的神经元对数
        return dsc_count / self.neuron_pair_num

    def dvc(self):
        dvc_count = 0
        for i in range(self.n - 1):  # 遍历前n-1层
            dc_cur = self.dc_dif_array[i]  # 第一层取dc
            vc_next = self.vc_dif_array[i + 1]  # 第二层取sc
            dvc_num = (vc_next[dc_cur == 1])
            np.append(self.dvc_arr, dvc_num)
            dvc_count_temp = np.sum(dvc_num)  # 第一层满足dc的用例,统计在第二层中满足sc的个数
            dvc_count += (dvc_count_temp * self.neuron_num_arr[i])  # 根据定义: 第二层发生变化的个数 * 当前层的神经元个数 = 总共发生变化的神经元对数
        return dvc_count / self.neuron_pair_num


if __name__ == '__main__':
    pass
