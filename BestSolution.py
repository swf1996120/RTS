import imagehash
import numpy as np
import torch
from keras_preprocessing.image import array_to_img
from tqdm import tqdm

from utils.model_conf import mnist, fashion

device = 1

def get_noise_threshold(data_name,sim_tolerate_num, x_train, y_train, x_train_hash):
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
        ssim = SSIM(n_channels=1, reduction='none').cuda(device)
    else:
        ssim = SSIM(n_channels=3, reduction='none').cuda(device)

    random_results = []
    for i in tqdm(range(len(random_indexs))):
        line_number, single_select, label = random_indexs[i], x_random_tensor[i], y_random[i]
        candidate_index = np.array(class_matrix[label])
        a, b = np.broadcast_arrays(x_train_hash[line_number], x_train_hash[candidate_index])
        a_hash_result = np.count_nonzero(a != b, axis=-1)
        hash_topk_value, hash_topk_indice = torch.topk(torch.tensor(a_hash_result), k=int(len(candidate_index) * 0.3),
                                                       dim=-1,
                                                       largest=False,
                                                       sorted=True)
        candidate_index = candidate_index[hash_topk_indice[1:]]  # 过滤自己

        img2 = torch.from_numpy(x_train[candidate_index]).float().permute(0, 3, 1, 2)
        img1, img2 = torch.broadcast_tensors(single_select, img2)
        if torch.cuda.is_available():
            img1 = img1.cuda(device)
            img2 = img2.cuda(device)
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
        ssim = SSIM(n_channels=1, reduction='none').cuda(device)
    else:
        ssim = SSIM(n_channels=3, reduction='none').cuda(device)

    select_tensor = torch.from_numpy(x_select).float().permute(0, 3, 1, 2)

    # 计算每个样本的ssim
    def _init_class_matrix():
        # 按照lable分类
        class_matrix = {}
        for i, lb in enumerate(y_train):
            if lb not in class_matrix:
                class_matrix[lb] = []
            class_matrix[lb].append(i)
        return class_matrix

    ssim_results = []
    class_matrix = _init_class_matrix()
    ssim_topk_values,ssim_topk_indices = [], []
    for line_number, single_select in tqdm(enumerate(select_tensor)):  # 测试用例的原始序列
        best_i, best_ssim = -1, -1
        for label, indexs in class_matrix.items():
            random_sample_number = int(np.min([len(indexs)*0.5, 50]))
            img2 = x_train[np.random.choice(indexs, random_sample_number, replace=False)]
            img2 = torch.from_numpy(img2).float().permute(0, 3, 1, 2)
            img1, img2 = torch.broadcast_tensors(single_select, img2)
            if torch.cuda.is_available():
                img1 = img1.cuda(device)
                img2 = img2.cuda(device)
            ssim_result = np.mean(ssim(img1, img2).data.cpu().numpy())
            if ssim_result > best_ssim:
                best_ssim = ssim_result
                best_i = label
        candidate_index = np.array(class_matrix[best_i])  # 获取的是对应的索引坐标
        # 获得对应的x_select的a_hash, 理论上hash值越小代表约相近
        a, b = np.broadcast_arrays(x_select_hash[line_number], x_train_hash[candidate_index])
        a_hash_result = np.count_nonzero(a != b, axis=-1)
        hash_topk_value, hash_topk_indice = torch.topk(torch.tensor(a_hash_result), k=int(len(candidate_index) * 0.3),
                                                       dim=-1,
                                                       largest=False,
                                                       sorted=True)
        candidate_index = candidate_index[hash_topk_indice]

        img2 = torch.from_numpy(x_train[candidate_index]).float().permute(0, 3, 1, 2)
        img1, img2 = torch.broadcast_tensors(single_select, img2)
        if torch.cuda.is_available():
            img1 = img1.cuda(device)
            img2 = img2.cuda(device)
        temp_result = ssim(img1, img2).data.cpu().numpy()
        ssim_result = np.zeros((len(x_train)))
        assert len(temp_result) == len(candidate_index)
        ssim_result[candidate_index] = temp_result
        value, indice = torch.topk(torch.from_numpy(ssim_result), k=int(sim_tolerate_num),
                                   dim=-1,
                                   largest=True,
                                   sorted=True)
        ssim_topk_values.append(value.reshape((1,-1)))
        ssim_topk_indices.append(indice.reshape((1,-1)))

    ssim_topk_values, ssim_topk_indices = torch.cat(ssim_topk_values, dim=0), torch.cat(ssim_topk_indices, dim=0)
    ssim_topk_values, ssim_topk_indices = ssim_topk_values.numpy(), ssim_topk_indices.numpy()

    return (ssim_topk_values, ssim_topk_indices)


def ob_sus_correct(sim_tolerate_num, ssim_datas,
                   select_pro, select_lable, train_pro, y_train, noise_threshold, filter_noise=False):
    '''
    ssim_results: ssim_scores
    ssim_topk_values: 前Top-k个
    ssim_topk_indices：对应的前Top_k的train的索引
    '''
    ssim_topk_values, ssim_topk_indices = ssim_datas

    ssim_1_values, ssim_1_indices = ssim_topk_values, ssim_topk_indices

    import torch.nn.functional as F

    y = torch.tensor(train_pro, dtype=torch.float32).unsqueeze(0).cuda()
    chuck = 200
    data_lists = torch.chunk(torch.tensor(select_pro, dtype=torch.float32).unsqueeze(1), chuck, dim=0)
    values, indices = [], []
    for data in data_lists:
        data = data.cuda()
        cosion_sim = F.cosine_similarity(data, y, dim=-1)
        value, indice = torch.topk(cosion_sim, k=int(sim_tolerate_num),
                                   dim=-1,
                                   largest=True,
                                   sorted=True)
        values.append(value.data.cpu())
        indices.append(indice.data.cpu())
        del data
    values, indices = torch.cat(values, dim=0), torch.cat(indices, dim=0)

    # -1:没有, 0:噪音, 1:correct, 2:wrong
    Tag = np.array([-1] * len(select_pro))
    # Line是select的指针
    for line_number in tqdm(range(len(indices))):
        # print(f"当前的序列为：{line_number}, 且此时的状态为{Tag[line_number]}, noise is {Tag}")
        if Tag[line_number] != -1:  # 已经有了定位
            continue
        if filter_noise:
            best_ssim = ssim_1_values[line_number]
            # 最相似的训练样本的索引
            best_index = ssim_1_indices[line_number]
            # 拥有相同相似训练样本的其他测试用例
            best_indexs = np.where(ssim_1_indices == best_index)[0]
            # 获得其他测试样本的最佳ssim值
            best_ssims = ssim_1_values[best_indexs]
            # 如果这个值等于best_ssim (考虑自身)
            # print(f"best_ssim is {best_ssim}, best_index is {best_index}, best_indexs is {best_indexs}, best_ssims is {best_ssims}, np.count_nonzero is {np.count_nonzero((best_ssims==best_ssim).astype(np.int))}")
            if np.count_nonzero((best_ssims == best_ssim).astype(np.int)) >= 2:
                for _ in best_indexs[best_ssims == best_ssim]:
                    if _ != line_number:
                        Tag[_] = 0

        if np.mean(ssim_topk_values[line_number]) < noise_threshold:  # 相似性很低
            Tag[line_number] = 0  # 添加到的噪音
        else:
            # 该测试用本和训练样本很相似
            best_sim_indexs = indices[line_number]
            no_noise_sim_indexs = list(filter(
                lambda index: select_lable[line_number] == y_train[index],
                best_sim_indexs))
            y_truth = y_train[ssim_topk_indices[line_number]]  # 对应的真实标签
            broad_list = np.broadcast_to(select_lable[line_number],
                                         y_truth.shape)
            judge = (broad_list == y_truth).astype(np.int)
            if np.count_nonzero(judge) >= len(judge) * 0.8 and len(no_noise_sim_indexs) >= len(best_sim_indexs) * 0.8:
                Tag[line_number] = 1
            else:
                Tag[line_number] = 2

    suspicious_array, correct_array, noise_array = np.where(Tag == 2)[0], np.where(Tag == 1)[0], np.where(Tag == 0)[0]
    assert np.all(Tag != -1)
    return suspicious_array, noise_array, correct_array


def get_max_coverage_index(statis_coverage):
    '''
    coverage, index, uncertain
    '''
    best_coverage = -1
    best_elements = []
    for i, coverage in enumerate(statis_coverage):
        if coverage > best_coverage:
            best_elements.clear()
            best_elements.append((i))
            best_coverage = coverage
        elif coverage == best_coverage:
            best_elements.append((i))
    return best_elements


def construct_empty_array(internal_number, nb_classes):
    coverage_array = np.zeros((nb_classes, internal_number))
    return coverage_array


def rank(candate_lists, internal_number, nb_classes):
    '''
    candidate_lists: array_index, index of sequence, uncertain
    '''
    total_length = len(candate_lists)
    rank = []
    coverage = construct_empty_array(internal_number, nb_classes)

    print(candate_lists.shape, candate_lists[0])

    def get_coverage(test):
        return nb_classes - np.count_nonzero(coverage[list(range(nb_classes)), test.astype(np.int)])

    def update_coverage(test):
        coverage[list(range(nb_classes)), test.astype(np.int)] = 1

    # array_index, index of whole sequence, uncertain
    for _ in tqdm(range(total_length)):
        statis_coverage = nb_classes - np.count_nonzero(
            coverage[list(range(nb_classes)), candate_lists[:, :-2].astype(np.int)], axis=-1)

        best_indexs = get_max_coverage_index(statis_coverage)  # 返回的是最大coverage的元素在candidate_list的索引位置

        if len(best_indexs) == len(candate_lists) and statis_coverage[0] - 0.0 <= 1e-6:
            coverage = construct_empty_array(internal_number, nb_classes)
            statis_coverage = nb_classes - np.count_nonzero(
                coverage[list(range(nb_classes)), candate_lists[:, :-2].astype(np.int)], axis=-1)

            best_indexs = get_max_coverage_index(statis_coverage)  # 返回的是最大coverage的元素在candidate_list的索引位置

        # 最大不确定度的
        uncertains = candate_lists[best_indexs][:, -2]  # 按照best_indexs的索引去获得不确定度值

        best_index = best_indexs[np.argmax(uncertains)]  # 获得在candidate_list的索引位置

        update_coverage(candate_lists[best_index][:-2])  # 更新

        rank.append(candate_lists[best_index][-1])

        candate_lists = np.delete(candate_lists, best_index, axis=0)

    return rank


def get_intervals(max_min_interval, num_intervals):
    def get_interval(length, num_interval):
        length_interval = length / num_interval  # 单个区间的长度
        interval_index = []
        for i in range(num_interval):
            interval_index.append(min + i * length_interval)
        return length_interval, interval_index

    result = []
    for intervals in max_min_interval:
        max, min = intervals[0], intervals[1]
        length_interval, interval_index = get_interval((max - min), num_intervals)
        assert len(interval_index) == num_intervals
        result.append((length_interval,interval_index))
    return result


def give_index(max_min_intervals, intervals, sample, i, keep):
    sample_index = np.zeros(len(sample))  # 暂时生成一个index数组
    for index in range(len(sample)):
        interval = intervals[index]  # 第i个特征的intervals
        interval_length, interval_index = interval[0], interval[1]  # 第i个特征范围 单位长度 以及索引位置
        for inner_index in range(len(interval_index)):  # 逐个查看索引区间
            limitation = max_min_intervals[index][0] if inner_index == len(interval_index) - 1 else \
                interval_index[
                    inner_index] + interval_length  # 如果 最后的一个索引
            if interval_index[inner_index] <= sample[index] <= limitation:
                sample_index[index] = inner_index
                break

    if not keep:  # 不保留主概率 将其概率值变为0
        sample_index[int(i)] = 0
    return sample_index


def Step_1(pred_test, pred_test_prob, num_parameter):
    max_min_interval = [(np.max(single_batch), np.min(single_batch)) for single_batch in pred_test_prob.T]

    # 根据设置的区间个数 求区间个数
    intervals = get_intervals(max_min_interval,
                              num_parameter)  # length_interval, [min + i for i in np.arange(0, length, length_interval)]

    # 得到每个样本的区间索引，越大索引概率越大
    all_indexs = [give_index(max_min_interval, intervals, pred_test_prob[i], pred_test[i], False)
                  for i in range(len(pred_test_prob))]

    return all_indexs


def get_uncertain(datas):
    uncertain = 1.0 - np.max(datas, axis=-1)

    return uncertain


def conquer(select_lable, select_pro, indexs, number_internals, nb_classes):
    all_indexs = np.array(Step_1(select_lable[indexs], select_pro[indexs], number_internals))
    # 获取索引+不确定度

    uncertains = get_uncertain(select_pro[indexs])  # array_index, index,  uncertain

    # print(f"all_indexs's shape {all_indexs.shape}, uncertains shape {uncertains.shape}")

    candidate_lists = np.concatenate([all_indexs, uncertains.reshape(-1, 1), np.array(indexs).reshape(-1, 1)], axis=-1)

    print(
        f"all_indexs's shape {all_indexs.shape}, uncertains shape {uncertains.shape}, candidate_lists shape {candidate_lists.shape}")

    scores = rank(candidate_lists, number_internals, nb_classes=nb_classes)

    return scores
