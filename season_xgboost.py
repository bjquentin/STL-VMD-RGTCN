from datetime import datetime
import os
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb
from data_preprocessing import mimaxscaler, train_test_step
from others import ia, tic, file_isnot_exist, get_eva
from lAOA import AOA
import global_var as gol

gol._init()


# def get_eva(scaler, model, X, y, type_test: bool = False):
#     predictions = np.zeros((y.shape[0], y.shape[1]))
#     inv_y_test = np.zeros((y.shape[0], y.shape[1]))
#
#     eva_test = np.zeros((7, y.shape[1]))
#     pre = model.predict(X)
#     for i in range(y.shape[1]):
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
#
#     if type_test:
#         data_array = np.hstack((inv_y_test, predictions))
#         data_df = pd.DataFrame(data_array)
#         data_df.columns = ['1-step-test', '2-step-test', '3-step-test',
#                            '1-step-prediction', '2-step-prediction', '3-step-prediction']
#
#         return eva_test, data_df
#     else:
#         return eva_test


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

    learning_rate, n_estimators = nest

    model = xgb.XGBRegressor(n_estimators=int(round(n_estimators)), reg_lambda=0.01,
                             learning_rate=learning_rate)
    model.fit(train_X, train_y)

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
        # eva.append([rmse, mape])
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
    print("\033[31mlearning_rate: {}, n_estimators: {} \033[0m".format(nest[0], int(round(nest[1]))))
    print("\033[32m{}\033[0m".format(df_eva))
    time2 = datetime.now()
    print("\033[31msprnd time: {}s\033[0m".format((time2 - time1).total_seconds()))
    print("\033[31m##############################################\033[0m")
    gol.set_value('fit_f_num', fit_f_num + 1)

    return sum_2


def get_nest():
    pop = 15
    MaxIter = 20
    dim = 2

    # learning_rate, gamma, reg_lambda
    lb = [0.001, 100]
    ub = [0.1, 300]

    fobj = fitness_function
    GbestScore, GbestPositon, Curve = AOA(pop, MaxIter, dim, lb, ub, fobj)

    # return optimal_value_
    return [GbestScore, GbestPositon]


def train_xgb(data: np.array, n_timestamp: int, loc: str, sp: pd.DataFrame = pd.DataFrame(), type_sp: bool = False):
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
        # learning_rate, gamma, reg_lambda
        nest = get_nest()
        s_p = [nest[0][0]]
        p = nest[1].flatten().tolist()
        s_p.append(p[0])
        s_p.append(p[1])
        s_p = np.array(s_p).reshape(1, -1)
        score_position = s_p.copy()
        end_t = datetime.now()
        s_t = (end_t - start_t).total_seconds()
        score_position = np.hstack((score_position, np.array(s_t).reshape(1, -1)))
        df_score_position = pd.DataFrame(score_position)
        df_score_position.columns = ['bestScore', 'learning_rate', 'n_estimators', 'spend_time']
        sp_path = f'dataset/position/{loc}/season/{loc}_xgb.csv'
        df_score_position.to_csv(sp_path, index=False)

        t = nest[1][0]
    learning_rate = t[0]
    n_estimators = t[1]

    xgb_model = xgb.XGBRegressor(n_estimators=int(round(n_estimators)), reg_lambda=0.01,
                                 learning_rate=learning_rate)
    xgb_model.fit(train_X, train_y)

    model_path = f'model/{loc}/season/{loc}_xgb.pkl'

    with open(model_path, "wb") as f:
        pickle.dump(xgb_model, f)

    x_y = {'train': [train_X, train_y], 'vali': [vali_X, vali_y], 'test': [test_X, test_y]}

    for tvt in ['train', 'vali', 'test']:
        data_x, data_y = x_y[tvt]
        df_eva, pre_prediction = get_eva(scaler[0], xgb_model, data_x, data_y, type_test=True)
        eva_path = f'dataset/evaluation/{loc}/{tvt}/season/{loc}_xgb_{tvt}.csv'
        pre_prediction_path = f'dataset/pre_ori/{loc}/{tvt}_prediction/season/{loc}_xgb.csv'
        df_eva.to_csv(eva_path, index=False)
        pre_prediction.to_csv(pre_prediction_path, index=False)

    return xgb_model, df_eva, df_score_position


def setup_seed(seed):
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


def get_model(loc: str, n_timestamp: int):
    path = f'dataset/stl/{loc}_season.csv'
    model_path = f'model/{loc}/season/{loc}_xgb.pkl'
    data = pd.read_csv(path).values[:, 1:]

    if file_isnot_exist(model_path):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        values = data.astype('float32')
        scaled, scaler = mimaxscaler(values)
        n_features = data.shape[1]

        train_x, train_y, vali_x, vali_y, test_x, test_y = train_test_step(scaled, n_timestamp, n_features, 3)

        train_X = train_x.reshape(-1, train_x.shape[1] * train_x.shape[2])
        vali_X = vali_x.reshape(-1, vali_x.shape[1] * vali_x.shape[2])
        test_X = test_x.reshape(-1, test_x.shape[1] * test_x.shape[2])

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        sp_path = f'dataset/position/{loc}/season/{loc}_xgb.csv'
        eva_test_path = f'dataset/evaluation/{loc}/test/season/{loc}_xgb_test.csv'
        if os.access(sp_path, os.F_OK):
            score_position = pd.read_csv(sp_path)
        else:
            score_position = pd.DataFrame()
        if os.access(eva_test_path, os.F_OK):
            eva_test = pd.read_csv(eva_test_path)
        else:
            eva_test, test_prediction = get_eva(scaler[0], model, test_X, test_y, True)

            eva_path = f'dataset/evaluation/{loc}/test/season/{loc}_xgb_test.csv'
            eva_test.to_csv(eva_path, index=False)

            test_prediction_path = f'dataset/pre_ori/{loc}/test_prediction/season/{loc}_xgb.csv'
            test_prediction.to_csv(test_prediction_path, index=False)

    else:
        sp_path = f'dataset/position/{loc}/season/{loc}_xgb.csv'
        if os.access(sp_path, os.F_OK):
            score_position = pd.read_csv(sp_path)
            model, eva_test, score_position = train_xgb(data, n_timestamp=n_timestamp, loc=loc,
                                                        sp=score_position, type_sp=True)
        else:
            model, eva_test, score_position = train_xgb(data, n_timestamp=n_timestamp, loc=loc)

    print(f'\033[32mscore_position:\n{score_position}\n eva_test:\n{eva_test}\033[0m')

    return model


def season_xgb(loc: str):
    path = f'dataset/stl/{loc}_season.csv'
    ret = os.access(path, os.F_OK)
    if ret:
        print(f'\033[32m{path} already exists\033[0m')
        timestamp = adfuller(pd.read_csv(f'dataset/air/{loc}_2022_nan.csv').AQI.values)[2]
        models = get_model(loc, timestamp)

    else:
        assert False, f'\033[31m{path} does not exist\033[0m'


if __name__ == '__main__':
    season_xgb('Beijing')
    season_xgb('Shanghai')
    season_xgb('Guangzhou')

