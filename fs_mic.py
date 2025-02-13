# Feature Selection
import numpy as np
from minepy import MINE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({
    "font.family": "sans-serif",  # specify font family here
    "font.sans-serif": ["Arial"],  # specify font here
    "font.size": 10})  # specify font size here

def print_stats(mine):
    print("MIC", mine.mic())
    print("MAS", mine.mas())
    print("MEV", mine.mev())
    print("MCN (eps=0)", mine.mcn(0))
    print("MCN (eps=1-MIC)", mine.mcn_general())
    print("GMIC", mine.gmic())
    print("TIC", mine.tic())


def get_mic(x: np.array, y: np.array) -> float:
    # mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine = MINE(alpha=0.6, c=20)
    mine.compute_score(x.reshape(-1), y.reshape(-1))
    return mine.mic()


def MIC_matirx(dataframe, mine):

    data = np.array(dataframe)
    n = len(data[0, :])
    result = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            mine.compute_score(data[:, i], data[:, j])
            result[i, j] = mine.gmic()
            result[j, i] = mine.gmic()
            # result[i, j] = mine.tic()
            # result[j, i] = mine.tic()
    RT = pd.DataFrame(result)
    return RT


def ShowHeatMap(DataFrame, label, data_name):
    DataFrame.index = label
    DataFrame.columns =label
    # colormap = plt.cm.RdYlBu
    # plt.figure(figsize=(14,12))
    # plt.title('Pearson Correlation of Features', y=1.05, size=15)
    # sns.heatmap(DataFrame.astype(float), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
    with plt.style.context(['nature', 'no-latex']):
        plt.figure(dpi=600)
        sns.heatmap(DataFrame, annot=True, linewidths=0.3, cmap="RdBu_r", fmt='.2f',
                    annot_kws={'size': 5}, vmax=1.0, vmin=0.0)
        plt.tight_layout()
        # plt.savefig('figure/corr_' + data_name + '.tiff', format='tiff', dpi=600, transparent=True, bbox_inches='tight')
        # plt.savefig('figure/corr_' + data_name + '.eps', format='eps', dpi=600, transparent=True, bbox_inches='tight')
        # plt.savefig('graph/corr_' + data_name + '.pdf', format='pdf', dpi=600, transparent=True)
        plt.show()
        a=1


def get_mean(data: pd.Series) -> pd.DataFrame:
    num = int(data.shape[0] / 3)
    arr = np.zeros((num))
    for i in range(num):
        mean = data.iloc[i * 3: (i + 1) * 3].mean()
        arr[i] = mean
    return pd.DataFrame(data=arr.reshape(-1, 1), columns=['AQI'])


def feature_selection(city: str = 'Beijing'):
    data_air_ = pd.read_csv(f'dataset/air/{city}_2022_nan.csv')

    def exception(data):
        data_exc = data.copy()
        over_index = data.AQI[data.AQI > 300].index
        for o_index in over_index:
            d_num = data.iloc[int(o_index - 24):int(o_index), 1:].mean().values
            data_exc.iloc[int(o_index), 1:] = d_num

        data_exc.AQI.plot()
        plt.tight_layout()
        plt.show()

        return data_exc

    a1 = exception(data_air_)
    data_air = exception(a1)

    mine = MINE(alpha=0.6, c=20)
    data_wine_mic_air = MIC_matirx(data_air.iloc[:, 1:], mine)
    print(data_wine_mic_air)
    data_wine_mic_air_df = pd.DataFrame(data_wine_mic_air, dtype=float)
    ShowHeatMap(data_wine_mic_air_df.copy(), data_air.columns[1:], city+'_air')

    data_wine_mic_air_ = data_wine_mic_air.iloc[0, :]
    data_wine_mic_air_.index = data_air.columns[1:]
    data_wine_mic_air_.sort_values(ascending=False, inplace=True)
    air_features = data_wine_mic_air_.index[1:3].tolist()
    air_features.insert(0, 'datetime')
    air_features.insert(1, 'AQI')
    data_air_features = data_air.loc[:, air_features].copy()


    data_weather = pd.read_csv(f'dataset/weather/{city}_weather_nan.csv')
    data_aqi = data_air.AQI
    # data_aqi = data_air.set_index('date_hour').loc[data_weather[['date_hour']].values.reshape(-1).tolist(), 'AQI']
    Data_a_w = pd.concat([data_aqi, data_weather.drop(labels=['datetime'], axis=1)], axis=1)
    data_wine_mic_we = MIC_matirx(Data_a_w, mine)
    print(data_wine_mic_we)
    data_wine_mic_we_df = pd.DataFrame(data_wine_mic_we, dtype=float)
    columns_str = ['Temp', 'Dew', 'Humi', 'Precip', 'Wind_s', 'Wind_g', 'Wind_d', 'Pressure', 'Solar_r']
    columns_str.insert(0, 'AQI')
    data_wine_mic_we_df.columns = Data_a_w.columns
    data_wine_mic_we_df.index = Data_a_w.columns
    print('corr:/n', Data_a_w.corr())
    ShowHeatMap(data_wine_mic_we_df.copy(), columns_str, city+'_weather')

    data_wine_mic_we_ = data_wine_mic_we_df.iloc[0, :]
    data_wine_mic_we_.sort_values(ascending=False, inplace=True)
    wether_features = data_wine_mic_we_.index[1:3].tolist()

    data_weather_feature = data_weather.loc[:, wether_features].copy()

    data = pd.concat([data_air_features, data_weather_feature], axis=1)

    df_feature_mic = pd.concat([data_wine_mic_air_df.iloc[0, 1:], data_wine_mic_we_df.iloc[0, 1:]])
    df_feature_mic.to_csv(f'dataset/{city}_feature_mic.csv', index=True, encoding='utf-8-sig')
    data.to_csv(f'dataset/{city}.csv', index=False)

    # set_seasonal(city)


if __name__ == '__main__':
    feature_selection(city='Beijing')
    feature_selection(city='Shanghai')
    feature_selection(city='Guangzhou')

