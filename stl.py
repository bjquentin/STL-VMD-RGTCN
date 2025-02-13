import numpy as np
import pandas
from statsmodels.tsa.seasonal import STL
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np


def stl_(data: pd.Series, n: int, n_timestamp: int) -> dict:
    stl = STL(data, period=n_timestamp * n)
    res = stl.fit()
    fig = res.plot()
    plt.tight_layout()
    plt.show()

    res.seasonal.plot()
    plt.tight_layout()
    plt.show()

    data_res = res.resid
    data_trend = res.trend
    data_seasonal = res.seasonal

    dict_stl = {'trend': data_trend, 'seasonal': data_seasonal, 'resid': data_res}

    return dict_stl


def get_stl(df_data: pd.DataFrame, loc: str, n: int = 5) -> int:
    result = adfuller(df_data.AQI.values)
    print(result)

    n_timestamp = result[2]  # 25
    print(n_timestamp)
    trend = df_data.iloc[:, 0].copy()
    seasonal = df_data.iloc[:, 0].copy()
    resid = df_data.iloc[:, 0].copy()

    columns = df_data.columns

    for i in range(1, df_data.shape[1]):
        dict_stl = stl_(df_data.iloc[:, i], n, n_timestamp)
        trend = pd.concat([trend, dict_stl['trend']], axis=1)
        seasonal = pd.concat([seasonal, dict_stl['seasonal']], axis=1)
        resid = pd.concat([resid, dict_stl['resid']], axis=1)
    trend.columns = columns
    seasonal.columns = columns
    resid.columns = columns
    trend.to_csv(f'dataset/stl/{loc}_trend.csv', index=False)
    seasonal.to_csv(f'dataset/stl/{loc}_season.csv', index=False)
    resid.to_csv(f'dataset/stl/{loc}_resid.csv', index=False)
    return n_timestamp


if __name__ == '__main__':
    # beijing n_timestamp 25
    beijing_path = f'dataset/Beijing.csv'
    beijing_data = pd.read_csv(beijing_path)
    beijing_timestamp = get_stl(beijing_data.copy(), loc='Beijing', n=2)

    # shanghai n_timestamp 28
    shanghai_path = f'dataset/Shanghai.csv'
    shanghai_data = pd.read_csv(shanghai_path)
    shanghai_timestamp = get_stl(shanghai_data.copy(), loc='Shanghai', n=2)

    # gunagzhou n_timestamp 32
    gunagzhou_path = f'dataset/Guangzhou.csv'
    gunagzhou_data = pd.read_csv(gunagzhou_path)
    gunagzhou_timestamp = get_stl(gunagzhou_data.copy(), loc='Guangzhou', n=3)
