import os

from cv2 import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

from tqdm import tqdm

import exp_utils
from exp_utils import mk_exp_dir
from gen_data.DauUtils import get_dau
from mop.mutation_op import Mop
from selection_method.nc_cover import get_cov_initer
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
from keras import backend as K
import numpy as np
from keras.engine.saving import load_model

'''
./exp_mutation_dataname_network/ps_data/TCPname_污染方式_0.npy:  测试用例排序算法的序列
'''


def get_probability(model_name, data_name, seed, restart):
    print(model_name, data_name)
    base_path = mk_exp_dir(exp_name, data_name, model_name, base_dir)
    exp(model_name, data_name, base_path, seed, restart)
    K.clear_session()


def get_noise_data(mop, x_dau, y_dau, data_name, seed=0):
    ratio = 1.0
    print("x_dau", len(x_dau), len(x_dau[0]))
    x_select = x_dau
    y_select = y_dau
    img_shape = x_select[0].shape
    x_add, y_add = mop.synthetic_data(len(x_select), img_shape, ratio=ratio, seed=seed)

    dau = exp_utils._get_IR_dau(data_name)
    (x_extra, _), (_, _) = dau.load_data(use_norm=True)
    if data_name == model_conf.caltech:
        x_extra = np.array([cv2.resize(i, (200, 200)) for i in x_extra])
    x_add_1, y_add_1 = mop.irrelevant_data(len(x_select), x_extra, ratio=ratio, seed=seed)
    x_add = np.concatenate((x_add, x_add_1), axis=0)
    y_add = np.concatenate((y_add, y_add_1), axis=0)
    del _
    x_add_1, y_add_1 = mop.corruption_data(x_select, y_select, ratio=ratio, seed=seed)
    x_add = np.concatenate((x_add, x_add_1), axis=0)
    y_add = np.concatenate((y_add, y_add_1), axis=0)

    x_add_1, y_add_1 = mop.repeat_data(x_select, y_select, ratio=ratio, seed=seed)
    x_add = np.concatenate((x_add, x_add_1), axis=0)
    y_add = np.concatenate((y_add, y_add_1), axis=0)

    #assert (y_add == -1).all()

    assert len(x_add) == len(y_add)
    return np.array(x_add), np.array(y_add)


def concat_df(df1, df2):
    if df1 is None:
        return np.array(df2)
    else:
        df = np.concatenate((df1, df2), axis=-1)
    return df


def count(orignal_data):
    result = None
    nb_class = 10
    print(f"orignal_data.shape is {orignal_data.shape}")
    for single in orignal_data:
        a = [0] * nb_class
        for value in single:
            transfor = value * 100
            index = int(transfor // 10)
            if index > 9:
                index = 9
            a[index] += 1
        assert np.sum(a) == len(single)
        result = concat_df(result, a)
    result = result.reshape((-1, nb_class))
    return result


def write2excel(base_path, orignal_data):
    output = open(os.path.join(base_path, 'count.xls'), 'w', encoding='gbk')
    output.write('correct_pro \t failure_pro \t noise_pro \n')  # 写完一行立马换行
    data = orignal_data
    for single in data:
        for value in single:
            output.write(str(value))  # write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')  # 相当于Tab一下，换一个单元格
        output.write('\n')  # 写完一行立马换行
    output.write('\n')  # 写完一行立马换行
    output.write('\n')  # 写完一行立马换行
    # 统计结果
    orignial_result = count(np.transpose(orignal_data, axes=(1, 0))) # 3*10

    result = np.transpose(orignial_result, axes=(1, 0)) # 10*3

    print(result.shape)

    for single in result:
        for value in single:
            output.write(str(value))  # write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')  # 相当于Tab一下，换一个单元格
        output.write('\n')  # 写完一行立马换行

    output.close()


def failure_test_inputs(dau, dau_name_arr):
    x_test_arr = []
    y_test_arr = []
    for dau_op_name in dau_name_arr:
        # print(dau_op_name)
        x, y = dau.load_dau_data(dau_op_name, use_norm=True, use_cache=False)
        x_test_arr.append(x[:])
        y_test_arr.append(y[:])
    return np.concatenate(x_test_arr,axis=0), np.concatenate(y_test_arr,axis=0)


# 重复数据实验
def exp(model_name, data_name, base_path, seed, restart):
    is_prepare_ps = True  # 是否需要再进行TCP算法的运行

    dau_name_arr = ['SF', 'ZM', 'BR', 'RT', 'CT', 'BL', 'SR']
    mop_name_arr = ["IR", "RG", "CR"]
    dau = get_dau(data_name)
    mop = Mop()
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)

    sample_num = 1
    model_path = model_conf.get_model_path(data_name, model_name)

    ############################
    # 实验
    ############################
    # 构造原始选择集

    print(f"The mop_name_arr is {mop_name_arr}")

    ori_model = load_model(model_path)

    if restart:

        failure_x, failure_y = failure_test_inputs(dau,dau_name_arr)

        noise_x,  noise_y = get_noise_data(mop, x_test, y_test, data_name, seed=0)

        judge = np.argmax(ori_model.predict(x_test),axis=-1) == y_test
        correct_x, correct_y = x_test[judge], y_test[judge]
        correct_index = np.random.choice(range(len(correct_x)),size=5000)
        correct_x,  correct_y = correct_x[correct_index], correct_y[correct_index]
        print(f"correct is {correct_x.shape}, {(np.argmax(ori_model.predict(correct_x),axis=-1) == correct_y).all()}")
        np.save(os.path.join(base_path, 'correct_x.npy'), correct_x)
        np.save(os.path.join(base_path, 'correct_y.npy'), correct_y)

        judge = np.argmax(ori_model.predict(failure_x),axis=-1) != failure_y
        failure_x, failure_y = failure_x[judge], failure_y[judge]
        failure_index = np.random.choice(range(len(failure_x)),size=5000)
        failure_x,  failure_y = failure_x[failure_index], failure_y[failure_index]
        print(f"failure is {failure_x.shape}, {(np.argmax(ori_model.predict(failure_x),axis=-1) != failure_y).all()}")
        np.save(os.path.join(base_path, 'failure_x.npy'), failure_x)
        np.save(os.path.join(base_path, 'failure_y.npy'), failure_y)


        noise_index = np.random.choice(range(len(noise_x)),size=5000)
        noise_x,  noise_y = noise_x[noise_index], noise_y[noise_index]
        np.save(os.path.join(base_path, 'noise_x.npy'), noise_x)
        np.save(os.path.join(base_path, 'noise_y.npy'), noise_y)

        assert len(np.unique(correct_y)) == 10 and len(np.unique(failure_y)) == 10

    else:
        correct_x, correct_y = np.load(os.path.join(base_path, 'correct_x.npy')), np.load(os.path.join(base_path, 'correct_y.npy'))
        failure_x, failure_y = np.load(os.path.join(base_path, 'failure_x.npy')), np.load(os.path.join(base_path, 'failure_y.npy'))
        noise_x, noise_y = np.load(os.path.join(base_path, 'noise_x.npy')), np.load(os.path.join(base_path, 'noise_y.npy'))

    print((np.argmax(ori_model.predict(correct_x),axis=-1)==correct_y).all())

    print((np.argmax(ori_model.predict(failure_x),axis=-1)!=failure_y).all())

    correct_pro = np.max(ori_model.predict(correct_x),axis=-1).reshape(len(correct_x),-1)
    failure_pro = np.max(ori_model.predict(failure_x),axis=-1).reshape(len(failure_x),-1)
    noise_pro   = np.max(ori_model.predict(noise_x),  axis=-1).reshape(len(noise_x),-1)
    pro_main = np.concatenate([correct_pro,failure_pro,noise_pro], axis=-1)

    write2excel(base_path, pro_main)


if __name__ == '__main__':
    from utils.model_conf import *

    '''
    '''
    base_dir = "motivation/"
    exp_name = "noise"
    model_data = {
        mnist: [LeNet5],
        svhn: [vgg16]
    }
    seed = 0
    restart = False
    for data_name, v_arr in tqdm(model_data.items()):
        for model_name in v_arr:
            get_probability(model_name, data_name, seed, restart)
