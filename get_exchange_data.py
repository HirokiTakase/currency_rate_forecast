#チャート取得系
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

#プロット系
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpl_finance
import matplotlib.dates as dates


from datetime import datetime
import json
import pandas as pd
import numpy as np
import dateutil.parser

def dt_to_str(dt):
    dt = float(dt)
    return datetime.fromtimestamp(dt)

def getexchangedata(start_day, end_day):

    # 認証キー
    api_key = '996fe3dfeb9dcb1ab55e9eccf5fd5d0c-b97b7549e373d6f7dd180689d4f6947e'
    api = API(access_token=api_key, environment="practice", headers={"Accept-Datetime-Format":"Unix"})
    # 1. データの取得開始日時を作成
    # start_day = datetime(2019, 11, 6)
    # end_day = datetime(2019, 11, 7)
    start = datetime.strftime(start_day, '%Y-%m-%dT%H:%M:%SZ')
    end = datetime.strftime(end_day, '%Y-%m-%dT%H:%M:%SZ')
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
    
    df.to_csv('usd_1min_trump.csv', encoding='utf-8',  index=False,)
   
    # 数値型に変換
    df["Date"] = mdates.date2num(df["Date"])
   
    

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
