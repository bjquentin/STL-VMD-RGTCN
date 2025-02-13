import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.saving.save import load_model

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller

import global_var as gol
from data_preprocessing import mimaxscaler, train_test_step
from lAOA import AOA
from nn.gru_tcn_ResNet import get_gtResNet
from others import file_isnot_exist, get_eva

gol._init()


# def get_eva(scaler, model, X, y):
#     predictions = np.zeros((y.shape[0], y.shape[1]))
#     inv_y_test = np.zeros((y.shape[0], y.shape[1]))
#
#     eva_test = np.zeros((7, y.shape[1]))
#     pre = model.predict(X)
#     for i in range(y.shape[1]):
#         # 进行预测
#         predictions[:, i] = scaler.inverse_transform(pre[:, i].reshape(-1, 1)).reshape(1, -1)
#         inv_y_test[:, i] = scaler.inverse_transform(y[:, i].reshape(-1, 1)).reshape(1, -1)
#
#         eva_test[0, i] = np.sqrt(mean_squared_error(inv_y_test[:, i], predictions[:, i]))  # RMSE
#         eva_test[1, i] = r2_score(inv_y_test[:, i], predictions[:, i])  # R2
#         eva_test[2, i] = mean_absolute_percentage_error(inv_y_test[:, i], predictions[:, i])  # MAPE
#         eva_test[3, i] = mean_absolute_error(inv_y_test[:, i], predictions[:, i])  # MAE
#         eva_test[4, i] = ia(inv_y_test[:, i], predictions[:, i])
#         eva_test[5, i] = tic(inv_y_test[:, i], predictions[:, i])
#         eva_test[6, i] = (inv_y_test[:, i] - predictions[:, i]).std()
#
#     eva_test = pd.concat(
#         [pd.DataFrame(['RMSE', 'R2', 'MAPE', 'MAE', 'IA', 'TIC', 'STD']), pd.DataFrame(eva_test)], axis=1)
#     eva_test.columns = ['Evaluation metrics', '1-step', '2-step', '3-step']
#     return eva_test


def cross_val(nest):
    scaler = gol.get_value('scaler')[0]
    scaled = gol.get_value('scaled')
    train_X = gol.get_value('train_X')
    train_y = gol.get_value('train_y')
    vali_X = gol.get_value('vali_X')
    vali_y = gol.get_value('vali_y')

    gru_units, conv_filters, conv_kernel_size, learning_rate, epochs = nest


    model = get_gtResNet(gru_units=int(round(gru_units)), conv_filters=int(round(conv_filters)),
                         conv_kernel_size=int(round(conv_kernel_size)), learning_rate=learning_rate,
                         input_shape=(train_X.shape[1], train_X.shape[2]), output_units=train_y.shape[1])

    model.fit(train_X, train_y, epochs=int(round(epochs)), batch_size=64, validation_data=(vali_X, vali_y), verbose=0)

    eva = []
    pre = model.predict(vali_X)
    for i in range(train_y.shape[1]):
        predictions = np.zeros((vali_X.shape[0], vali_y.shape[1]))
        inv_y_vali = np.zeros((vali_X.shape[0], vali_y.shape[1]))
        predictions[:, i] = scaler.inverse_transform(pre[:, i].reshape(-1, 1)).reshape(1, -1)
        inv_y_vali[:, i] = scaler.inverse_transform(vali_y[:, i].reshape(-1, 1)).reshape(1, -1)
        rmse = np.sqrt(mean_squared_error(inv_y_vali[:, i], predictions[:, i]))
        mape = mean_absolute_percentage_error(inv_y_vali[:, i], predictions[:, i]) * 100
        r2 = r2_score(inv_y_vali[:, i], predictions[:, i])
        eva.append([rmse, r2])
    return eva


def fitness_function(nest):
    def get_weight(data: np.array):
        list_data = list(data)
        list_1 = [1 / i for i in list_data]
        list_2 = [i / sum(list_1) for i in list_1]
        return list_2

    time1 = datetime.now()
    eva = cross_val(nest)
    df_eva = pd.DataFrame(eva).T
    df_eva.index = ['RMSE', 'r2']
    df_eva.columns = ['step-1', 'step-2', 'step-3']
    print("\033[31m{}\033[0m".format(datetime.now()))

    fit_f_num = gol.get_value('fit_f_num')

    df_eva_ = df_eva.copy()
    arr_eva = df_eva_.values
    r2_1 = (1 - arr_eva[1, :]) * 10
    arr_eva[1, :] = r2_1

    sum_0 = df_eva_.sum(axis=0)
    weight_0 = get_weight(sum_0.values)
    df_eva_0 = df_eva_ * weight_0
    sum_1 = df_eva_0.sum(axis=1)
    weight_1 = get_weight(sum_1.values)
    df_eva_1 = (sum_1.T * weight_1).T
    sum_2 = df_eva_1.sum(axis=0)

    print("\033[31mfitness_function {}: {} \033[0m".format(fit_f_num, sum_2))
    print("\033[31mgru_units: {}, conv_filters: {}, conv_kernel_size: {}, learning_rate: {}, epochs: {} \033[0m"
          .format(int(round(nest[0])), int(round(nest[1])), int(round(nest[2])), nest[3], int(round(nest[4]))))
    print("\033[32m{}\033[0m".format(df_eva))
    time2 = datetime.now()
    print("\033[31msprnd time: {}s\033[0m".format((time2 - time1).total_seconds()))
    print("\033[31m############################################################################################\033[0m")
    gol.set_value('fit_f_num', fit_f_num + 1)

    return sum_2


def get_nest():
    pop = 15
    MaxIter = 20
    dim = 5

    # gru_units, conv_filters, conv_kernel_size, learning_rate, epochs
    lb = [1, 1, 1, 0.0001, 1]
    ub = [100, 100, 10, 0.01, 100]

    fobj = fitness_function
    GbestScore, GbestPositon, Curve = AOA(pop, MaxIter, dim, lb, ub, fobj)
    # print(optimal_value_)

    # return optimal_value_
    return [GbestScore, GbestPositon]


def train_gru_tcn(data: np.array, n_timestamp: int, loc: str, imf_num: int, sp: pd.DataFrame = pd.DataFrame(),
                  type_sp: bool = False, type_res: int = 2):
    if type_res == 0:
        sp_path = f'dataset/position/{loc}/{loc}_gru_tcn_stl_res.csv'
        model_path = f'model/{loc}/{loc}_gru_tcn_stl_res.h5'

    elif type_res == 1:
        sp_path = f'dataset/position/{loc}/{loc}_gru_tcn.csv'
        model_path = f'model/{loc}/{loc}_gru_tcn.h5'

    else:
        sp_path = f'dataset/position/{loc}/resid/gru_tcn/{loc}_gru_tcn_imf{imf_num}.csv'
        model_path = f'model/{loc}/resid/gru_tcn/{loc}_gru_tcn_imf{imf_num}.h5'

    fit_f_num = 1
    start_t = datetime.now()
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    values = data.astype('float32')
    scaled, scaler = mimaxscaler(values)

    gol.set_value('fit_f_num', fit_f_num)
    gol.set_value('scaler', scaler)
    gol.set_value('scaled', scaled)

    n_features = data.shape[1]

    train_x, train_y, vali_x, vali_y, test_x, test_y = train_test_step(scaled, n_timestamp, n_features, 3)

    gol.set_value('train_X', train_x)
    gol.set_value('train_y', train_y)
    gol.set_value('vali_X', vali_x)
    gol.set_value('vali_y', vali_y)
    gol.set_value('test_X', test_x)
    gol.set_value('test_y', test_y)

    if type_sp:
        t = sp.values.tolist()[0][1:]
        df_score_position = sp.copy()
    else:
        nest = get_nest()
        s_p = [nest[0][0]]
        p = nest[1].flatten().tolist()
        s_p.append(int(round(p[0])))
        s_p.append(int(round(p[1])))
        s_p.append(int(round(p[2])))
        s_p.append(p[3])
        s_p.append(int(round(p[4])))
        s_p = np.array(s_p).reshape(1, -1)
        score_position = s_p.copy()

        end_t = datetime.now()
        s_t = (end_t - start_t).total_seconds()
        score_position = np.hstack((score_position, np.array(s_t).reshape(1, -1)))
        df_score_position = pd.DataFrame(score_position)
        df_score_position.columns = ['bestScore', 'gru_units', 'conv_filters', 'conv_kernel_size',
                                     'learning_rate', 'epochs', 'spend_time']

        df_score_position.to_csv(sp_path, index=False)

        t = nest[1][0]
    gru_units = int(round(t[0]))
    conv_filters = int(round(t[1]))
    conv_kernel_size = int(round(t[2]))
    learning_rate = t[3]
    epochs = int(round(t[4]))

    model = get_gtResNet(gru_units=int(round(gru_units)), conv_filters=int(round(conv_filters)),
                         conv_kernel_size=int(round(conv_kernel_size)), learning_rate=learning_rate,
                         input_shape=(train_x.shape[1], train_x.shape[2]), output_units=train_y.shape[1])

    model.fit(train_x, train_y, epochs=int(round(epochs)), batch_size=64, validation_data=(test_x, test_y), verbose=1)

    model.save(model_path)

    x_y = {'train': [train_x, train_y], 'vali': [vali_x, vali_y], 'test': [test_x, test_y]}
    for tvt in ['train', 'vali', 'test']:
        if type_res == 0:
            eva_path = f'dataset/evaluation/{loc}/{tvt}/{loc}_gru_tcn_stl_res_{tvt}.csv'
            pre_prediction_path = f'dataset/pre_ori/{loc}/{tvt}_prediction/{loc}_gru_tcn_stl_res.csv'
        elif type_res == 1:
            eva_path = f'dataset/evaluation/{loc}/{tvt}/{loc}_gru_tcn_{tvt}.csv'
            pre_prediction_path = f'dataset/pre_ori/{loc}/{tvt}_prediction/{loc}_gru_tcn.csv'

        else:
            eva_path = f'dataset/evaluation/{loc}/{tvt}/resid/gru_tcn/{loc}_gru_tcn_imf{imf_num}_{tvt}.csv'
            pre_prediction_path = f'dataset/pre_ori/{loc}/{tvt}_prediction/resid/gru_tcn/{loc}_gru_tcn_imf{imf_num}.csv'
        # eva_path = f'dataset/evaluation/{loc}/{tvt}/resid/gru_tcn/{loc}_gru_tcn_imf{imf_num}_{tvt}.csv'
        # pre_prediction_path = f'dataset/pre_ori/{loc}/{tvt}_prediction/resid/gru_tcn/{loc}_gru_tcn_imf{imf_num}.csv'

        data_x, data_y = x_y[tvt]
        df_eva, pre_prediction = get_eva(scaler[0], model, data_x, data_y, type_test=True)
        df_eva.to_csv(eva_path, index=False)
        pre_prediction.to_csv(pre_prediction_path, index=False)


    # df_eva, test_prediction = get_eva(scaler[0], model, test_x, test_y, type_test=True)
    #
    # df_eva.to_csv(eva_path, index=False)
    #
    # test_prediction.to_csv(test_prediction_path, index=False)

    return model, df_eva, df_score_position


def setup_seed(seed):
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


def get_model(loc: str, n_timestamp: int, imf_num: int = 0, res: int = 2):
    if res == 0:
        path = f'dataset/stl/{loc}_resid.csv'
        model_path = f'model/{loc}/{loc}_gru_tcn_stl_res.h5'
        sp_path = f'dataset/position/{loc}/{loc}_gru_tcn_stl_res.csv'
        eva_path = f'dataset/evaluation/{loc}/test/{loc}_gru_tcn_stl_res_test.csv'
        test_prediction_path = f'dataset/pre_ori/{loc}/test_prediction/{loc}_gru_tcn_stl_res.csv'
        data = pd.read_csv(path).iloc[:, 1:]
    elif res == 1:
        path = f'dataset/{loc}.csv'
        model_path = f'model/{loc}/{loc}_gru_tcn.h5'
        sp_path = f'dataset/position/{loc}/{loc}_gru_tcn.csv'
        eva_path = f'dataset/evaluation/{loc}/test/{loc}_gru_tcn_test.csv'
        test_prediction_path = f'dataset/pre_ori/{loc}/test_prediction/{loc}_gru_tcn.csv'
        data = pd.read_csv(path).iloc[:, 1:]
    else:
        path = f'dataset/decom/vmd/{loc}_imf{imf_num}.csv'
        model_path = f'model/{loc}/resid/gru_tcn/{loc}_gru_tcn_imf{imf_num}.h5'
        sp_path = f'dataset/position/{loc}/resid/gru_tcn/{loc}_gru_tcn_imf{imf_num}.csv'
        eva_path = f'dataset/evaluation/{loc}/test/resid/gru_tcn/{loc}_gru_tcn_imf{imf_num}_test.csv'
        test_prediction_path = f'dataset/pre_ori/{loc}/test_prediction/resid/gru_tcn/{loc}_gru_tcn_imf{imf_num}.csv'
        data = pd.read_csv(path)

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    values = data.values.astype('float32')
    scaled, scaler = mimaxscaler(values)

    n_features = data.shape[1]

    train_x, train_y, vali_x, vali_y, test_x, test_y = train_test_step(scaled, n_timestamp, n_features, 3)

    if file_isnot_exist(model_path):
        model = load_model(model_path)
        if os.access(sp_path, os.F_OK):
            score_position = pd.read_csv(sp_path)
        else:
            score_position = pd.DataFrame()

        if os.access(eva_path, os.F_OK):
            eva_test = pd.read_csv(eva_path)
        else:

            eva_test, test_prediction = get_eva(scaler[0], model, test_x, test_y, type_test=True)

            eva_test.to_csv(eva_path, index=False)

            test_prediction.to_csv(test_prediction_path, index=False)

    else:
        if os.access(sp_path, os.F_OK):
            score_position = pd.read_csv(sp_path)
            model, eva_test, score_position = train_gru_tcn(data.values, n_timestamp=n_timestamp, loc=loc,
                                                            imf_num=imf_num, sp=score_position, type_sp=True,
                                                            type_res=res)
        else:
            model, eva_test, score_position = train_gru_tcn(data.values, n_timestamp=n_timestamp,
                                                            loc=loc, imf_num=imf_num, type_res=res)

    print(f'\033[32m score_position:\n{score_position}\n eva_test:\n{eva_test}\033[0m')

    return model


def resid_gru_tcn(loc: str, type_res: int = 2):
    if type_res == 0:
        path_stl_resid = f'dataset/stl/{loc}_resid.csv'
        ret = os.access(path_stl_resid, os.F_OK)
        if ret:
            print(f'\033[32m {path_stl_resid} already exists\033[0m')
            timestamp = adfuller(pd.read_csv(f'dataset/air/{loc}_2022_nan.csv').AQI.values)[2]
            models = get_model(loc, timestamp, res=type_res)
        else:
            assert False, f'\033[31m{path_stl_resid} does not exist\033[0m'
    elif type_res == 1:
        path_data = f'dataset/{loc}.csv'
        ret = os.access(path_data, os.F_OK)
        if ret:
            print(f'\033[32m {path_data} already exists\033[0m')
            timestamp = adfuller(pd.read_csv(f'dataset/air/{loc}_2022_nan.csv').AQI.values)[2]
            models = get_model(loc, timestamp, res=type_res)
        else:
            assert False, f'\033[31m{path_data} does not exist\033[0m'
    else:
        path = f'dataset/decom/{loc}/{loc}_AQI_vmd.csv'
        imf_num = pd.read_csv(path).shape[1] - 1
        # imf_num = 5
        for i in range(imf_num):
        # for i in [7]:
            path_imf = f'dataset/decom/vmd/{loc}_imf{i + 1}.csv'
            ret = os.access(path_imf, os.F_OK)
            if ret:
                print(f'\033[32m {path_imf} already exists\033[0m')
                timestamp = adfuller(pd.read_csv(f'dataset/air/{loc}_2022_nan.csv').AQI.values)[2]
                models = get_model(loc, timestamp, i + 1)

            else:
                assert False, f'\033[31m{path} does not exist\033[0m'


if __name__ == '__main__':
    setup_seed(42)
    resid_gru_tcn('Beijing')
    resid_gru_tcn('Shanghai')
    resid_gru_tcn('Guangzhou')

    # resid_gru_tcn('Beijing', 0)
    # resid_gru_tcn('Beijing', 1)
    # resid_gru_tcn('Shanghai', 0)
    # resid_gru_tcn('Shanghai', 1)
    # resid_gru_tcn('Guangzhou', 0)
    # resid_gru_tcn('Guangzhou', 1)
