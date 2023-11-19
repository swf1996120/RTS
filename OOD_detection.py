import os
import random

import hdbscan
import imagehash
import pandas as pd
import torch
from keras_preprocessing.image import array_to_img
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

from utils.model_conf import mnist, fashion, LeNet5

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

from tqdm import tqdm
from keras.engine.saving import load_model
import exp_utils
from gen_data.DauUtils import get_dau
from mop.mutation_op import Mop
from utils import model_conf
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需动态分配显存
session = tf.Session(config=config)
import matplotlib.pyplot as plt

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(0)
plt.switch_backend('agg')
from keras import backend as K, Model
import os
import numpy as np

'''
./exp_mutation_dataname_network/ps_data/TCPname_污染方式_0.npy:  测试用例排序算法的序列
'''


def RTS_OOD(dataName, x_select, x_train, y_train):
    def get_noise_threshold(data_name, sim_tolerate_num, x_train, y_train, x_train_hash):
        # 计算每个样本的ssim
        def _init_class_matrix():
            # 按照lable分类
            class_matrix = {}
            for i, lb in enumerate(y_train):
                if lb not in class_matrix:
                    class_matrix[lb] = []
                class_matrix[lb].append(i)
            return class_matrix

        class_matrix = _init_class_matrix()

        random_indexs = np.random.choice(len(x_train), int(len(x_train) * 0.1), replace=False)

        x_random = x_train[random_indexs]
        y_random = y_train[random_indexs]

        x_random_tensor = torch.from_numpy(x_random).float().permute(0, 3, 1, 2)
        from piqa import SSIM
        if data_name in [mnist, fashion]:
            ssim = SSIM(n_channels=1, reduction='none').cuda(1)
        else:
            ssim = SSIM(n_channels=3, reduction='none').cuda(1)

        random_results = []
        for i in tqdm(range(len(random_indexs))):
            line_number, single_select, label = random_indexs[i], x_random_tensor[i], y_random[i]
            candidate_index = np.array(class_matrix[label])
            a, b = np.broadcast_arrays(x_train_hash[line_number], x_train_hash[candidate_index])
            a_hash_result = np.count_nonzero(a != b, axis=-1)
            hash_topk_value, hash_topk_indice = torch.topk(torch.tensor(a_hash_result),
                                                           k=int(len(candidate_index) * 0.3),
                                                           dim=-1,
                                                           largest=False,
                                                           sorted=True)
            candidate_index = candidate_index[hash_topk_indice[1:]]  # 过滤自己

            img2 = torch.from_numpy(x_train[candidate_index]).float().permute(0, 3, 1, 2)
            img1, img2 = torch.broadcast_tensors(single_select, img2)
            if torch.cuda.is_available():
                img1 = img1.cuda(1)
                img2 = img2.cuda(1)
            temp_result = ssim(img1, img2).data.cpu()
            ssim_topk_values, ssim_topk_indices = torch.topk(temp_result, k=sim_tolerate_num, dim=-1,
                                                             largest=True,
                                                             sorted=True)
            random_results.append(torch.mean(ssim_topk_values).numpy())
        random_results = np.array(random_results)
        return np.min(random_results) + 0.05

    def obtain_ssim(data_name, sim_tolerate_num, x_select, x_train, y_train, x_select_hash, x_train_hash):
        # 通过SSIM过滤噪音

        from piqa import SSIM
        if data_name in [mnist, fashion]:
            ssim = SSIM(n_channels=1, reduction='none').cuda(1)
        else:
            ssim = SSIM(n_channels=3, reduction='none').cuda(1)

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        select_tensor = torch.from_numpy(x_select).float().permute(0, 3, 1, 2).to(device)
        x_train_tensor = torch.from_numpy(x_train).float().permute(0, 3, 1, 2).to(device)

        # 计算每个样本的ssim
        def _init_class_matrix():
            # 按照lable分类
            class_matrix = {}
            for i, lb in enumerate(y_train):
                if lb not in class_matrix:
                    class_matrix[lb] = []
                class_matrix[lb].append(i)
            return class_matrix

        class_matrix = _init_class_matrix()

        ssim_topk_values, ssim_topk_indices = [], []
        for line_number, single_select in tqdm(enumerate(select_tensor)):  # 测试用例的原始序列
            best_i, best_ssim = -1, -1
            for label, indexs in class_matrix.items():
                random_sample_number = int(np.min([len(indexs) * 0.5, 50]))
                img2_indexs = np.random.choice(indexs, random_sample_number, replace=False)
                img2_tensor = x_train_tensor[img2_indexs]
                img1, img2 = torch.broadcast_tensors(single_select, img2_tensor)
                ssim_result = torch.mean(ssim(img1, img2))
                if ssim_result > best_ssim:
                    best_ssim = ssim_result
                    best_i = label
            candidate_index = np.array(class_matrix[best_i])  # 获取的是对应的索引坐标
            # 获得对应的x_select的a_hash, 理论上hash值越小代表约相近
            a, b = np.broadcast_arrays(x_select_hash[line_number], x_train_hash[candidate_index])
            a_hash_result = np.count_nonzero(a != b, axis=-1)
            hash_topk_value, hash_topk_indice = torch.topk(torch.tensor(a_hash_result).to(device),
                                                           k=int(len(candidate_index) * 0.3),
                                                           dim=-1,
                                                           largest=False,
                                                           sorted=True)
            candidate_index = candidate_index[hash_topk_indice.cpu()]

            img2 = x_train_tensor[candidate_index]
            img1, img2 = torch.broadcast_tensors(single_select, img2)
            temp_result = ssim(img1, img2).data.cpu().numpy()
            ssim_result = np.zeros((len(x_train)))
            assert len(temp_result) == len(candidate_index)
            ssim_result[candidate_index] = temp_result
            value, indice = torch.topk(torch.from_numpy(ssim_result), k=int(sim_tolerate_num),
                                       dim=-1,
                                       largest=True,
                                       sorted=True)
            ssim_topk_values.append(value.reshape((1, -1)))
            ssim_topk_indices.append(indice.reshape((1, -1)))

        ssim_topk_values, ssim_topk_indices = torch.cat(ssim_topk_values, dim=0), torch.cat(ssim_topk_indices, dim=0)
        ssim_topk_values, ssim_topk_indices = ssim_topk_values.numpy(), ssim_topk_indices.numpy()

        return (ssim_topk_values, ssim_topk_indices)

    def ob_noise(ssim_datas, noise_threshold):

        ssim_topk_values, ssim_topk_indices = ssim_datas

        Tag = np.array([-1] * len(ssim_topk_values))
        # Line是select的指针
        for line_number in tqdm(range(len(ssim_topk_values))):
            if np.mean(ssim_topk_values[line_number]) < noise_threshold:  # 相似性很低
                Tag[line_number] = 0  # 添加到的噪音

        noise_array = np.where(Tag == 0)[0]

        return noise_array

    x_select_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_select])
    x_train_hash = np.array([imagehash.phash(array_to_img(i)).hash.flatten() for i in x_train])

    noise_threshold = get_noise_threshold(data_name=dataName, sim_tolerate_num=5, x_train=x_train,
                                          y_train=y_train, x_train_hash=x_train_hash)

    ssim_datas = obtain_ssim(data_name=dataName, sim_tolerate_num=5, x_select=x_select, x_train=x_train,
                             y_train=y_train,
                             x_select_hash=x_select_hash,
                             x_train_hash=x_train_hash)

    noise_array = ob_noise(ssim_datas=ssim_datas, noise_threshold=noise_threshold)

    return noise_array


def HDBSCAN_OOD(model_name, data_name, x_select, n_components=8, min_cluster_size=80, min_samples=4):
    sub_layer_index = -1

    if model_name in [model_conf.LeNet1, model_conf.resNet20]:
        sub_layer_index = -2
    elif model_name in [model_conf.LeNet5, model_conf.vgg16]:
        sub_layer_index = -4

    orig_model = load_model(model_conf.get_model_path(datasets=data_name, model_name=model_name))

    dense_layer_model = Model(inputs=orig_model.input,
                              outputs=orig_model.layers[sub_layer_index].output)  # 获取到某一层的layer子模型

    dense_layer_model.summary()

    dense_output = dense_layer_model.predict(x_select)

    minMax = MinMaxScaler()  # MinMaxScaler
    dense_output = minMax.fit_transform(dense_output)

    if model_name in [model_conf.LeNet1, model_conf.LeNet5, model_conf.resNet20, model_conf.vgg16]:
        from sklearn.decomposition import FastICA
        fica = FastICA(n_components=n_components)
        dense_output = fica.fit_transform(dense_output)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    r = clusterer.fit(dense_output)

    labels = r.labels_

    print(f" max label is {np.max(labels)} and the min label is {np.min(labels)}")

    label_noise = []
    for i, l in enumerate(labels):
        if l == -1:
            label_noise.append(i)

    return np.array(label_noise)


def MSP_OOD(model_name, data_name, x_select, x_train):
    def compute_threshold(x_train, orig_model):
        scores = np.max(orig_model.predict(x_train), axis=1)
        # 获取scores的最小值作为threshold
        threshold = np.min(scores)

        return threshold

    orig_model = load_model(model_conf.get_model_path(datasets=data_name, model_name=model_name))

    threshold = compute_threshold(x_train, orig_model)

    # 获取x_select的预测结果
    scores = np.max(orig_model.predict(x_select), axis=1)

    # 获取噪音的index
    noise_index = np.where(scores < threshold)[0]

    return noise_index


def One_class_svm_OOD(model_name, data_name, x_train, x_select, nu=0.1):
    sub_layer_index = -1

    if model_name in [model_conf.LeNet1, model_conf.resNet20]:
        sub_layer_index = -2
    elif model_name in [model_conf.LeNet5, model_conf.vgg16]:
        sub_layer_index = -4

    orig_model = load_model(model_conf.get_model_path(datasets=data_name, model_name=model_name))

    dense_layer_model = Model(inputs=orig_model.input,
                              outputs=orig_model.layers[sub_layer_index].output)  # 获取到某一层的layer子模型

    dense_layer_model.summary()

    x_select_embedding = dense_layer_model.predict(x_select).reshape(x_select.shape[0], -1)

    x_train_embedding = dense_layer_model.predict(x_train).reshape(x_train.shape[0], -1)

    # 训练模型
    clf = OneClassSVM(gamma='scale', nu=nu)
    clf.fit(x_train_embedding)

    # 预测 unlabeled_data
    pred = clf.predict(x_select_embedding)

    # 获取噪音数据的索引
    noise_idx = [i for i, p in enumerate(pred) if p == -1]

    return np.array(noise_idx)


def Embedding_similarity(model_name, data_name, x_select, x_train, block_size=1024):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def compute_threshold(x_train_embeddings, k=5):
        with torch.no_grad():
            x_train_embeddings = torch.tensor(x_train_embeddings, device=device)
            n = x_train_embeddings.size(0)
            sample_size = max(n // 10, k + 1)
            sample_idx = random.sample(range(n), sample_size)
            sample_embeddings = x_train_embeddings[sample_idx]
            dist = torch.cdist(sample_embeddings, x_train_embeddings, p=1)
            top_k_dist, _ = torch.topk(dist, k + 1, dim=1, largest=False, sorted=True)
            threshold = torch.max(torch.mean(top_k_dist[:, 1:], dim=1))
        return threshold.item()

    sub_layer_index = -1

    if model_name in [model_conf.LeNet1, model_conf.resNet20]:
        sub_layer_index = -2
    elif model_name in [model_conf.LeNet5, model_conf.vgg16]:
        sub_layer_index = -4

    orig_model = load_model(model_conf.get_model_path(datasets=data_name, model_name=model_name))

    dense_layer_model = Model(inputs=orig_model.input,
                              outputs=orig_model.layers[sub_layer_index].output)  # 获取到某一层的layer子模型

    dense_layer_model.summary()

    x_select_embeddings = dense_layer_model.predict(x_select)

    x_train_embeddings = dense_layer_model.predict(x_train)

    threshold = compute_threshold(x_train_embeddings=x_train_embeddings)

    noisy_idx_blocks = []
    with torch.no_grad():
        x_select_embeddings = torch.tensor(x_select_embeddings, device=device)
        x_train_embeddings = torch.tensor(x_train_embeddings, device=device)
        for i in range(0, len(x_select_embeddings), block_size):
            x_block = x_select_embeddings[i:i + block_size]
            dist = torch.cdist(x_block, x_train_embeddings, p=1)
            top_k_dist, _ = torch.topk(dist, 5, dim=1, largest=False, sorted=True)
            mean_dist = torch.mean(top_k_dist, dim=1)
            is_noisy = (mean_dist > threshold)
            noisy_idx_blocks.append(torch.nonzero(is_noisy).flatten() + i)

    # 将分块内的索引转换为整个数据集上的索引
    noisy_idx = torch.cat(noisy_idx_blocks).cpu().numpy()

    return noisy_idx


def ood_query(ood_method, dataName, model_name, x_select, x_train, y_train):
    if ood_method == "Embedding_similarity":
        return Embedding_similarity(model_name, dataName, x_select, x_train, block_size=1024)
    elif ood_method == "One_class_svm_OOD":
        return One_class_svm_OOD(model_name, dataName, x_train, x_select, nu=0.1)
    elif ood_method == "HDBSCAN_OOD":
        return HDBSCAN_OOD(model_name, dataName, x_select, n_components=8, min_cluster_size=80, min_samples=4)
    elif ood_method == "RTS_OOD":
        return RTS_OOD(dataName, x_select, x_train, y_train)
    elif ood_method == "MSP_OOD":
        return MSP_OOD(model_name, dataName, x_select, x_train)
    return None


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def cal_metrics(y_select, noisy_total_index, noise_idex):
    # 将噪声样本标记为1，非噪声样本标记为0
    y_select_bin = [1 if i in noisy_total_index else 0 for i in range(len(y_select))]
    noise_index_bin = [1 if i in noise_idex else 0 for i in range(len(y_select))]
    accuracy = accuracy_score(y_select_bin, noise_index_bin)
    precision = precision_score(y_select_bin, noise_index_bin)
    recall = recall_score(y_select_bin, noise_index_bin)
    f1 = f1_score(y_select_bin, noise_index_bin)
    auc = roc_auc_score(y_select_bin, noise_index_bin)
    return accuracy, precision, recall, f1, auc


def exec(model_name, data_name, seed):
    # 实验
    print(model_name, data_name)
    base_path = exp_utils.mk_exp_dir("ood", data_name, model_name, base_dir)
    exp(model_name, data_name, base_path, seed)
    K.clear_session()


def obtain_confusion_matrix(y_select, noise_index):
    # 初始化混淆矩阵
    tp = fp = tn = fn = 0
    # 统计混淆矩阵
    for i in range(len(y_select)):
        if y_select[i] == -1:
            if i in noise_index:
                tp += 1
            else:
                fn += 1
        else:
            if i in noise_index:
                fp += 1
            else:
                tn += 1
    return tp, tn, fp, fn


# 重复数据实验
def exp(model_name, data_name, base_path, seed):
    dau_name_arr = exp_utils.dau_name_arr  # 扩增方法
    mop_name_arr = ["IR", "RG", "CR"]
    dau = get_dau(data_name)
    mop = Mop()
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)

    split_ratio = 0.5
    sample_num = 1

    is_clear_retrain = False  # 是否需要将重新训练的结果清空

    ############################
    # 实验
    ############################

    # 构造原始选择集
    x_dau_test_arr, y_dau_test_arr, x_val_dict, y_val_dict = exp_utils.get_dau_data(x_test, y_test, dau, dau_name_arr,
                                                                                    ratio=split_ratio,
                                                                                    shuffle=True)

    for mop_name in mop_name_arr:  # ["IR","RG","CR"]   # 所谓的纯洁的数据集就是原始的数据集，其他的数据集都额外的添加了一些数据
        for i in range(sample_num):  # 每个算子5次,每次的数据组成都不一样
            if mop_name == "ORI":
                x_select = np.concatenate(x_dau_test_arr, axis=0)
                y_select = np.concatenate(y_dau_test_arr, axis=0)
            else:
                x_select, y_select = exp_utils.get_mutation_data(mop_name, mop, x_dau_test_arr, y_dau_test_arr,
                                                                 data_name, seed=seed, )  # seed 是相同的，所以产生的污染数据是相同的
            # -1标志为噪声数据，
            noisy_total_index = np.where(y_select == -1)[0]

            ps_csv_path = os.path.join(base_path, "{}_{}.csv".format("OOD", mop_name))

            ood_methods = ["Embedding_similarity", "One_class_svm_OOD", "HDBSCAN_OOD", "RTS_OOD", "MSP_OOD"]

            if is_clear_retrain:
                results_df = pd.DataFrame(
                    columns=["OOD Method", "Accuracy", "Precision", "Recall", "F1-score", "AUC", "TP", "TN", "FP", "FN",
                             "Detected Noise Count",
                             "Total Noise Count"])
            else:
                try:
                    results_df = pd.read_csv(ps_csv_path)
                except FileNotFoundError:
                    results_df = pd.DataFrame(
                        columns=["OOD Method", "Accuracy", "Precision", "Recall", "F1-score", "AUC", "TP", "TN", "FP",
                                 "FN",
                                 "Detected Noise Count",
                                 "Total Noise Count"])

            for ood_method in ood_methods:

                if (results_df is not None) and (ood_method in results_df['OOD Method'].values):
                    continue

                noise_index = ood_query(ood_method, dataName=data_name, model_name=model_name, x_select=x_select,
                                        x_train=x_train, y_train=y_train)

                accuracy, precision, recall, f1, auc = cal_metrics(y_select, noisy_total_index, noise_index)
                tp, tn, fp, fn = obtain_confusion_matrix(y_select=y_select, noise_index=noise_index)

                detected_noise_count = len(noise_index)
                total_noise_count = len(noisy_total_index)
                results_df = results_df.append(
                    {"OOD Method": ood_method,
                     "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1, "AUC": auc,
                     "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                     "Detected Noise Count": detected_noise_count, "Total Noise Count": total_noise_count},
                    ignore_index=True)

            results_df.to_csv(ps_csv_path, index=False)

        del x_select


if __name__ == '__main__':

    parameters = [("", 0)]

    for name, seed in parameters:
        base_dir = "result_raw/OOD{}".format(name)
        for data_name, v_arr in tqdm(model_conf.model_data.items()):
            for model_name in v_arr:
                exec(model_name, data_name, seed)
