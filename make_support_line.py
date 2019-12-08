import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# ローソク足描写
import mpl_finance
import matplotlib.pyplot as plt
# import matplotlib.finance as mpf
from mpl_finance import candlestick2_ohlc

import tqdm
import subprocess
# low : 終値（pandas series）
# high : 高値（pandas series）
# n：S/Rライン検出の対象とするローソク足の本数
# min_touches：S/Rラインの検出の基となるローソク足との接触数
# stat_likeness_percent：ライン検出のエラーマージン（許容度）
# bounce_percent：反転時のレート値動きの割合

# sup：サポートラインのレート
# res：レジスタンスラインのレート
# sup_break：サポートラインの反転時
# res_break：レジスタンスラインの反転時

def supres(low, high, n=28, min_touches=2, stat_likeness_percent=1.5, bounce_percent=5):
 
    df = pd.concat([high, low], keys = ['high', 'low'], axis=1)
    df['sup'] = pd.Series(np.zeros(len(low)))
    df['res'] = pd.Series(np.zeros(len(low)))
    df['sup_break'] = pd.Series(np.zeros(len(low)))
    df['sup_break'] = 0
    df['res_break'] = pd.Series(np.zeros(len(high)))
    df['res_break'] = 0
    
    for x in range((n-1)+n, len(df)):
        print(x)
        subprocess.call('clear')
        tempdf = df[x-n:x+1].copy()
        sup = None
        res = None
        maxima = tempdf.high.max()
        minima = tempdf.low.min()
        move_range = maxima - minima
        move_allowance = move_range * (stat_likeness_percent / 100)
        bounce_distance = move_range * (bounce_percent / 100)
        touchdown = 0
        awaiting_bounce = False
        for y in range(0, len(tempdf)):
            
            if abs(maxima - tempdf.high.iloc[y]) < move_allowance and not awaiting_bounce:
                touchdown = touchdown + 1
                awaiting_bounce = True
            elif abs(maxima - tempdf.high.iloc[y]) > bounce_distance:
                awaiting_bounce = False
        if touchdown >= min_touches:
            res = maxima
 
        touchdown = 0
        awaiting_bounce = False
        for y in range(0, len(tempdf)):
            if abs(tempdf.low.iloc[y] - minima) < move_allowance and not awaiting_bounce:
                touchdown = touchdown + 1
                awaiting_bounce = True
            elif abs(tempdf.low.iloc[y] - minima) > bounce_distance:
                awaiting_bounce = False
        if touchdown >= min_touches:
            sup = minima
        if sup:
            df['sup'].iloc[x] = sup
        if res:
            df['res'].iloc[x] = res
    res_break_indices = list(df[(np.isnan(df['res']) & ~np.isnan(df.shift(1)['res'])) & (df['high'] > df.shift(1)['res'])].index)
    for index in res_break_indices:
        df['res_break'].at[index] = 1
    sup_break_indices = list(df[(np.isnan(df['sup']) & ~np.isnan(df.shift(1)['sup'])) & (df['low'] < df.shift(1)['sup'])].index)
    for index in sup_break_indices:
        df['sup_break'].at[index] = 1
    ret_df = pd.concat([df['sup'], df['res'], df['sup_break'], df['res_break']], keys = ['sup', 'res', 'sup_break', 'res_break'], axis=1)
    return ret_df

def main():

    # データの読み込み
    df = pd.read_csv("usd_1min2019.csv")
    
    # トレンドラインを作成するための対象データ
    ranges = slice(56721,57021,None)

    # トレンドラインデータ作成
    levels = supres(df['Low'][ranges],
                    df['High'][ranges],
                    n=50,
                    min_touches=5,
                    stat_likeness_percent=10,
                    bounce_percent=3)
 
    levels[(levels['sup'] > 0) | (levels['res'] > 0)][0:5]

    fig = plt.figure(figsize=(18, 9))
    ax = plt.subplot(1, 1, 1)
    candle_temp = df[ranges]
    candle_temp = candle_temp.reset_index()
    candlestick2_ohlc(
        ax, candle_temp["Open"], candle_temp["High"], candle_temp["Low"],
        candle_temp["Close"], width=0.9, colorup="r", colordown="b"
    )
    
    for sup in levels['sup']:
        if sup != np.nan:
            ax.axhline(y=sup, linewidth='1.0', color='red')
    for res in levels['res']:
        if res != np.nan:
            ax.axhline(y=res, linewidth='1.0', color='blue')
    fig.tight_layout()
    plt.savefig('images/save.png')

if __name__ == '__main__':
    main()