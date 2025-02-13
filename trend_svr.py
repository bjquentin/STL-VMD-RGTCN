import os
import pickle
from datetime import datetime

import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller

import global_var as gol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from data_preprocessing import data_split_tvt, get_auto_corr, mimaxscaler, train_test_step
from lAOA import AOA
from others import ia, tic, file_isnot_exist, get_eva

gol._init()

# def get_eva(scaler, models, ori_X, ori_y):
#     predictions = np.zeros((ori_X.shape[0], ori_y.shape[1]))
#     inv_y_test = np.zeros((ori_X.shape[0], ori_y.shape[1]))
#
#     eva_test = np.zeros((7, ori_y.shape[1]))
#
#     for m in range(len(models)):
#         model = models[m]
#         predictions[:, m] = scaler.inverse_transform(model.predict(ori_X).reshape(-1, 1)).reshape(1, -1)
#         inv_y_test[:, m] = scaler.inverse_transform(ori_y[:, m].reshape(-1, 1)).reshape(1, -1)
#
#         eva_test[0, m] = np.sqrt(mean_squared_error(inv_y_test[:, m], predictions[:, m]))  # RMSE
#         eva_test[1, m] = r2_score(inv_y_test[:, m], predictions[:, m])  # R2
#         eva_test[2, m] = mean_absolute_percentage_error(inv_y_test[:, m], predictions[:, m])  # MAPE
#         eva_test[3, m] = mean_absolute_error(inv_y_test[:, m], predictions[:, m])  # MAE
#         eva_test[4, m] = ia(inv_y_test[:, m], predictions[:, m])
#         eva_test[5, m] = tic(inv_y_test[:, m], predictions[:, m])
#         eva_test[6, m] = (inv_y_test[:, m] - predictions[:, m]).std()
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

    x_train = []
    y_train = []

    x_verify = []
    y_verify = []

    c, eps, gamma = nest

    models = []
    for i in range(train_y.shape[1]):
        # svr_model = SVR(kernel='rbf', C=c, epsilon=eps)
        svr_model = SVR(kernel='rbf', C=c, gamma=gamma, epsilon=eps)
        svr_model.fit(train_X, train_y[:, i])
        models.append(svr_model)

    predictions = np.zeros((vali_X.shape[0], vali_y.shape[1]))
    inv_y_vali = np.zeros((vali_X.shape[0], vali_y.shape[1]))

    eva = []
    for m in range(len(models)):
        model = models[m]
        predictions[:, m] = scaler.inverse_transform(model.predict(vali_X).reshape(-1, 1)).reshape(1, -1)
        inv_y_vali[:, m] = scaler.inverse_transform(vali_y[:, m].reshape(-1, 1)).reshape(1, -1)
        rmse = np.sqrt(mean_squared_error(inv_y_vali[:, m], predictions[:, m]))
        mape = mean_absolute_percentage_error(inv_y_vali[:, m], predictions[:, m]) * 100
        eva.append([rmse, mape])

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
    df_eva.index = ['RMSE', 'MAPE']
    df_eva.columns = ['step-1', 'step-2', 'step-3']
    print("\033[31m{}\033[0m".format(datetime.now()))

    fit_f_num = gol.get_value('fit_f_num')

    sum_0 = df_eva.sum(axis=0)
    weight_0 = get_weight(sum_0.values)
    df_eva_0 = df_eva * weight_0
    sum_1 = df_eva_0.sum(axis=1)
    weight_1 = get_weight(sum_1.values)
    df_eva_1 = (sum_1.T * weight_1).T
    sum_2 = df_eva_1.sum(axis=0)

    print("\033[31mfitness_function {}: {} \033[0m".format(fit_f_num, sum_2))
    print("\033[31mC: {}, epsilon: {} \033[0m".format(nest[0], nest[1]))
    print("\033[32m{}\033[0m".format(df_eva))
    time2 = datetime.now()
    print("\033[31msprnd time: {}s\033[0m".format((time2 - time1).total_seconds()))
    print("\033[31m##############################################\033[0m")
    gol.set_value('fit_f_num', fit_f_num + 1)

    return sum_2


def get_nest():
    pop = 15
    MaxIter = 20
    dim = 3
    # C, gamma, eps
    lb = [1, 0.01, 0.01]
    ub = [100, 1, 1]
    fobj = fitness_function
    GbestScore, GbestPositon, Curve = AOA(pop, MaxIter, dim, lb, ub, fobj)

    # return optimal_value_
    return [GbestScore, GbestPositon]


def train_svr(data: np.array, n_timestamp: int, loc: str, sp: pd.DataFrame = pd.DataFrame(), type_sp: bool = False):
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

    train_X = train_x.reshape(-1, train_x.shape[1] * train_x.shape[2])
    vali_X = vali_x.reshape(-1, vali_x.shape[1] * vali_x.shape[2])
    test_X = test_x.reshape(-1, test_x.shape[1] * test_x.shape[2])

    gol.set_value('train_X', train_X)
    gol.set_value('train_y', train_y)
    gol.set_value('vali_X', vali_X)
    gol.set_value('vali_y', vali_y)
    gol.set_value('test_X', test_X)
    gol.set_value('test_y', test_y)

    if type_sp:
        t = sp.values.tolist()[0][1:]
        df_score_position = sp.copy()
    else:
        nest = get_nest()
        s_p = [nest[0][0]]
        p = nest[1].flatten().tolist()
        s_p.append(p[0])
        s_p.append(p[1])
        s_p.append(p[2])
        s_p = np.array(s_p).reshape(1, -1)
        score_position = s_p.copy()
        end_t = datetime.now()
        s_t = (end_t - start_t).total_seconds()
        score_position = np.hstack((score_position, np.array(s_t).reshape(1, -1)))
        df_score_position = pd.DataFrame(score_position)
        df_score_position.columns = ['bestScore', 'C', 'epsilon', 'gamma', 'spend_time']
        sp_path = f'dataset/position/{loc}/trend/{loc}_svr.csv'
        # df_score_position.to_csv(sp_path, index=False)
        t = nest[1][0]

    c = t[0]
    eps = t[1]
    gamma = t[2]
    models = []
    for i in range(train_y.shape[1]):
        model_path = f'model/{loc}/trend/{loc}_svr_step_{i + 1}.pkl'
        svr_model = SVR(kernel='rbf', C=c, gamma=gamma, epsilon=eps)
        svr_model.fit(train_X, train_y[:, i])

        with open(model_path, "wb") as f:
            pickle.dump(svr_model, f)

        models.append(svr_model)

    x_y = {'train': [train_X, train_y], 'vali': [vali_X, vali_y], 'test': [test_X, test_y]}
    for tvt in ['train', 'vali', 'test']:
        data_x, data_y = x_y[tvt]
        df_eva, pre_prediction = get_eva(scaler[0], models, data_x, data_y, type_test=True, type_svr=True)
        eva_path = f'dataset/evaluation/{loc}/{tvt}/trend/{loc}_svr_{tvt}.csv'
        pre_prediction_path = f'dataset/pre_ori/{loc}/{tvt}_prediction/trend/{loc}_svr.csv'
        df_eva.to_csv(eva_path, index=False)
        pre_prediction.to_csv(pre_prediction_path, index=False)

    # df_eva, test_prediction = get_eva(scaler[0], models, test_X, test_y, type_test=True, type_svr=True)
    #
    # eva_path = f'dataset/evaluation/trend/{loc}_svr_test.csv'
    # df_eva.to_csv(eva_path, index=False)
    #
    # test_prediction_path = f'dataset/pre_ori/test_prediction/trend/{loc}_svr.csv'
    # test_prediction.to_csv(test_prediction_path, index=False)

    return models, df_eva, df_score_position


def setup_seed(seed):
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


def get_model(loc: str, n_timestamp: int):
    path = f'dataset/stl/{loc}_trend.csv'
    model_path = [f'model/{loc}/trend/{loc}_svr_step_1.pkl', f'model/{loc}/trend/{loc}_svr_step_2.pkl',
                  f'model/{loc}/trend/{loc}_svr_step_3.pkl']
    data = pd.read_csv(path).values[:, 1:]

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    values = data.astype('float32')
    scaled, scaler = mimaxscaler(values)
    n_features = data.shape[1]

    train_x, train_y, vali_x, vali_y, test_x, test_y = train_test_step(scaled, n_timestamp, n_features, 3)

    train_X = train_x.reshape(-1, train_x.shape[1] * train_x.shape[2])
    vali_X = vali_x.reshape(-1, vali_x.shape[1] * vali_x.shape[2])
    test_X = test_x.reshape(-1, test_x.shape[1] * test_x.shape[2])

    if file_isnot_exist(model_path[0]) and file_isnot_exist(model_path[1]) and file_isnot_exist(model_path[2]):
        models = []
        for mp in model_path:
            with open(mp, "rb") as f:
                models.append(pickle.load(f))
        sp_path = f'dataset/position/{loc}/trend/{loc}_svr.csv'
        eva_test_path = f'dataset/evaluation/{loc}/trend/{loc}_svr_test.csv'
        score_position = pd.read_csv(sp_path)
        if os.access(eva_test_path, os.F_OK):
            eva_test = pd.read_csv(eva_test_path)
        else:
            eva_test, test_prediction = get_eva(scaler[0], models, test_X, test_y, type_test=True, type_svr=True)
            eva_path = f'dataset/evaluation/{loc}/test/trend/{loc}_svr_test.csv'
            eva_test.to_csv(eva_path, index=False)

            test_prediction_path = f'dataset/pre_ori/{loc}/test_prediction/trend/{loc}_svr.csv'
            test_prediction.to_csv(test_prediction_path, index=False)
    else:
        data = pd.read_csv(path).iloc[:, 1:]
        sp_path = f'dataset/position/{loc}/trend/{loc}_svr.csv'
        if os.access(sp_path, os.F_OK):
            score_position = pd.read_csv(sp_path)
            models, eva_test, score_position = train_svr(data.values, n_timestamp=n_timestamp,
                                                         loc=loc, sp=score_position, type_sp=True)
        else:
            models, eva_test, score_position = train_svr(data.values, n_timestamp=n_timestamp, loc=loc)

    print(f'\033[32m score_position:\n{score_position}\n eva_test:\n{eva_test}\033[0m')

    return models


def trend_svr(loc: str):

    path = f'dataset/stl/{loc}_trend.csv'
    ret = os.access(path, os.F_OK)
    if ret:
        print(f'\033[32m{path} already exists\033[0m')
        timestamp = adfuller(pd.read_csv(f'dataset/air/{loc}_2022_nan.csv').AQI.values)[2]
        models = get_model(loc, timestamp)
    else:
        assert False, f'\033[31m{path} does not exist\033[0m'


if __name__ == '__main__':
    trend_svr('Beijing')
    trend_svr('Shanghai')
    trend_svr('Guangzhou')