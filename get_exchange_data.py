from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

import json
import pandas as pd    


def dt_to_str(dt):
    if dt is None:
        return ''
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def getexchangedata():

    # 認証キー
    api_key = '996fe3dfeb9dcb1ab55e9eccf5fd5d0c-b97b7549e373d6f7dd180689d4f6947e'
    api = API(access_token=api_key, environment="practice", headers={"Accept-Datetime-Format":"Unix"})
    
    # 探索設定
    params = {
    'count': 100,
    'granularity': 'M1',
    'endtime': '2019-11-06T00:00:00.000000Z',
    }

    # リクエスト
    r = instruments.InstrumentsCandles(instrument='USD_JPY', params=params)
    response = api.request(r)
    print(json.dumps(response, indent=2))
    

    data = []
    for raw in r.response['candles']:
        data.append([raw['time'], raw['volume'], raw['mid']['o'], raw['mid']['h'], raw['mid']['l'], raw['mid']['c']])
    
    df = pd.DataFrame(data)
    df.columns = ['time', 'volume', 'open', 'high', 'low', 'close']
    df = df.set_index('time')
    df.index = dt_to_str(df.index)
    print(df.tail())


if __name__ == '__main__':
    getexchangedata()