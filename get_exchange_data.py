#チャート取得系
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

#プロット系
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpl_finance
import matplotlib.dates as dates


from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

import make_weekday_array as mwa
import os
import sys
def dt_to_str(dt):
    dt = float(dt)
    return datetime.fromtimestamp(dt)

def getexchangedata(start_time, end_time):
    # 認証キー
    api_key = '996fe3dfeb9dcb1ab55e9eccf5fd5d0c-b97b7549e373d6f7dd180689d4f6947e'
    api = API(access_token=api_key, environment="practice", headers={"Accept-Datetime-Format":"Unix"})
    # 1. データの取得開始日時を作成
    # start_time = datetime(2017, 1, 1)
    # end_time = datetime(2017, 1, 31)
    
    #過去データ取得
    if os.path.exists('./usd_1min7.csv') == True:
        df_old = pd.read_csv('./usd_1min7.csv')
        df_old['Date'] = pd.to_datetime(df_old['Date'])
        start_time =  pd.to_datetime(df_old.iloc[-1]["Date"])+ timedelta(minutes = 1) 
        # end_time = datetime.now() -timedelta(minutes = 1) - timedelta(hours = 9) 
        marge_flg = True
    else :
        marge_flg = False
        pass
    # start_time =  datetime(2010,1,2)
    
    # 取得データを表示
    print("SEARCH START TIME : ", start_time)
    print("SEARCH END TIME   : "  ,end_time)
    
    weekDay_df = mwa.make_weekday_array(start_time,end_time, True)

    cnt = 0
    for i in weekDay_df:

        start = datetime.strftime(weekDay_df['Start'][cnt], '%Y-%m-%dT%H:%M:%SZ')
        end = datetime.strftime(weekDay_df['End'][cnt], '%Y-%m-%dT%H:%M:00Z')

        # 探索設定
        params = {
        #'count': 10,指定するときは使えない
        'granularity': 'M1',
        'from': start,
        'to': end,
        'alignmentTimezone': 'Japan'
        }
        # リクエスト
        r = instruments.InstrumentsCandles(instrument='USD_JPY', params=params)
        response = api.request(r)
        #データ形成
        data = []
        for raw in r.response['candles']:
            data.append([raw['time'], raw['volume'], raw['mid']['o'], raw['mid']['h'], raw['mid']['l'], raw['mid']['c']])

        cnt = cnt + 1

    df = pd.DataFrame(data)
    df.columns = ['Date', 'volume', 'Open', 'High', 'Low', 'Close']
    df = df.drop(['volume'], axis = 1)
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    
    # datetimeに変換
    cnt = 0
    for t in df['Date']:
        t = dt_to_str(t)
        df.loc[cnt,'Date'] = t
        cnt = cnt + 1

    if marge_flg == True:
        df = pd.concat([df_old, df])
    
    df.to_csv('./usd_1min7.csv', encoding='utf-8',  index=False,)
   
    # 数値型に変換
    # df["Date"] = mdates.date2num(df["Date"])
   
    

    # # fig = plt.figure()

    # ax = plt.subplot(1, 1, 1)
    # # candlestickを使って描画
    # mpl_finance.candlestick_ohlc(ax, df[["Date", "Open", "High", "Low", "Close"]].values, width=(1/24/60)*0.1, colorup="b", colordown="r")
    # # 時系列用のLocator/Formatterを使用
    # ax.grid()
    # ax.xaxis.set_major_locator(mdates.MonthLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    # plt.tight_layout()
    # plt.savefig('candlestick_day.png')
    # plt.show()
    
    return True

if __name__ == '__main__':

    getexchangedata(datetime(2019,7,13,20,37,0), None)
