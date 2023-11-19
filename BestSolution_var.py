import numpy as np
import numpy as np
import torch
from tqdm import tqdm



def ob_sus_correct_var1(sim_tolerate_num, ssim_datas,
                        select_pro, select_lable, train_pro, y_train):
    '''
    没有过滤噪音的步骤
    '''
    ssim_topk_values, ssim_topk_indices = ssim_datas

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

    suspicious_array, correct_array = np.where(Tag == 2)[0], np.where(Tag == 1)[0]
    assert np.all(Tag != -1)
    return suspicious_array, correct_array


def ob_sus_correct_var2(ssim_datas, select_pro, noise_threshold):
    '''
    没有过滤正确测试用例的步骤
    '''
    ssim_topk_values, ssim_topk_indices = ssim_datas

    # -1:没有, 0:噪音, 1:correct, 2:wrong
    Tag = np.array([-1] * len(select_pro))
    # Line是select的指针
    for line_number in tqdm(range(len(select_pro))):
        if Tag[line_number] != -1:  # 已经有了定位
            continue
        if np.mean(ssim_topk_values[line_number]) < noise_threshold:  # 相似性很低
            Tag[line_number] = 0  # 添加到的噪音
        else:
            Tag[line_number] = 2

    suspicious_array, noise_array = np.where(Tag == 2)[0], np.where(Tag == 0)[0]
    assert np.all(Tag != -1)
    return suspicious_array, noise_array


def conquer(indexs):
    '''
    随机排
    '''
    scores = indexs
    np.random.shuffle(scores)
    return scores.tolist()
