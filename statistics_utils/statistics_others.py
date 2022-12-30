import os
import keras
from keras.engine.saving import load_model
from tqdm import tqdm
from exp_utils import get_dau_data
from utils import model_conf
from gen_data.DauUtils import get_dau
import numpy as np
from statistics_utils.statistics_utils import get_base_table_dir
from utils.utils import add_df


def RQ_acc_base():
    dau_name_arr = ['SF', 'ZM', 'BR', 'RT', 'CT', 'BL', 'SR']  # 'BU', 'BD' # 扩增方法
    split_ratio = 0.5
    nb_classes = model_conf.fig_nb_classes
    df = None
    for data_name, v_arr in tqdm(model_conf.model_data.items()):
        for model_name in v_arr:
            csv_data = {}
            dau = get_dau(data_name)
            (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
            # dup_ratio_arr = np.arange(0.3, 0.41, 0.1)
            model_path = model_conf.get_model_path(data_name, model_name)
            ori_model = load_model(model_path)

            # 构造原始选择集
            x_dau_test_arr, y_dau_test_arr, x_val_dict, y_val_dict = get_dau_data(x_test, y_test, dau, dau_name_arr,
                                                                                  ratio=split_ratio,
                                                                                  shuffle=True)

            x_val_arr = []
            y_val_arr = []
            for op, x_val in x_val_dict.items():
                y_val = y_val_dict[op]
                x_val_arr.append(x_val)
                y_val_arr.append(y_val)

            X_val = np.concatenate(x_val_arr, axis=0)
            Y_val = np.concatenate(y_val_arr, axis=0)
            Y_val_vec = keras.utils.np_utils.to_categorical(Y_val, nb_classes)
            acc_base = ori_model.evaluate(X_val, Y_val_vec)[1]
            csv_data["data_name"] = data_name
            csv_data["model_name"] = model_name
            csv_data["acc_base"] = acc_base
            # print(csv_data)
            df = add_df(df, csv_data)
    tab_dir = get_base_table_dir()
    os.makedirs(tab_dir, exist_ok=True)
    tab_path = os.path.join(tab_dir, "acc_ori.csv")
    df.to_csv(tab_path)
