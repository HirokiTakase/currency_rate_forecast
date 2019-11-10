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

def getexchangedata():

    # 認証キー
    api_key = '996fe3dfeb9dcb1ab55e9eccf5fd5d0c-b97b7549e373d6f7dd180689d4f6947e'
    api = API(access_token=api_key, environment="practice", headers={"Accept-Datetime-Format":"Unix"})
    # 1. データの取得開始日時を作成
    start_day = datetime(2019, 11, 6)
    end_day = datetime(2019, 11, 7)
    start = datetime.strftime(start_day, '%Y-%m-%dT%H:%M:%SZ')
    end = datetime.strftime(end_day, '%Y-%m-%dT%H:%M:%SZ')
    # 探索設定
    params = {
    #'count': 10,指定するときは使えない
    'granularity': 'M1',
    'from': start,
    'to': end
    }
    # リクエスト
    r = instruments.InstrumentsCandles(instrument='USD_JPY', params=params)
    response = api.request(r)
    #データ形成
    data = []
    for raw in r.response['candles']:
        data.append([raw['time'], raw['volume'], raw['mid']['o'], raw['mid']['h'], raw['mid']['l'], raw['mid']['c']])
    
    df = pd.DataFrame(data)
    df.columns = ['time', 'volume', 'open', 'high', 'low', 'close']
    exchange_data = df.drop(['volume'], axis = 1)
    print(exchange_data['time'].dtype)
    exchange_data.set_index('time', inplace=True)
    cols = exchange_data.select_dtypes(exclude=['float']).columns
    exchange_data[cols] = exchange_data[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    
    

    
    # cnt = 0
    # for t in exchange_data['time']:
    #    t = dt_to_str(t)
    #    exchange_data.loc[cnt,'time'] = t
    #    cnt = cnt + 1
    df_ = exchange_data.copy()
    df_ = df_.reset_index()
    df_['time'] = df_['time'].astype(float)/2150
    
    
    
    # df_.index = mdates.date2num(df_.index.to_pydatetime())
   
    align_data = df_.values
    print(align_data)
   

    
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    mpl_finance.candlestick_ohlc(ax, align_data, width=2, alpha=0.5, colorup='r', colordown='b')
    ax.grid()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))

    plt.savefig('candlestick_day.png')

    

if __name__ == '__main__':
    getexchangedata()