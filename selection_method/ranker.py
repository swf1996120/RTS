from typing import List

import faiss
import fast_pytorch_kmeans
import imagehash
import keras
import mkl
import torch
from keras import Model
from keras import backend as K
from keras_preprocessing.image import array_to_img
from tqdm import tqdm

import BestSolution1
import BestSolution_filter
import BestSolution_var
import OOD_detection
import general_util
import tf_util
from ATS.ATS import ATS
from BestSolution import obtain_ssim, conquer, ob_sus_correct, get_noise_threshold
from selection_method.necov_method import metrics
from selection_method.rank_method.CES.condition import CES_ranker
from utils import model_conf

mkl.get_max_threads()


class Ranker(object):
    def __init__(self, model, x):
        self.model = model
        self.x = x

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


