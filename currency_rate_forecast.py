import pandas as pd 
import os
import numpy as np
import mpl_finance
import matplotlib.pyplot as plt
# import matplotlib.finance as mpf
from mpl_finance import candlestick_ohlc
from datetime import datetime, timedelta
import datetime
 
# # 自然言語処理（NLP）を扱うライブラリ
import nltk
from nltk.corpus import stopwords
import string
 
# # 機械学習ライブラリ
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
 
import get_tweet_data as gtd
import get_exchange_data as ged

# # 設定
pd.options.display.max_colwidth = -1

def main():
    # 実行ファイルの絶対パス取得
    os.chdir("/Users/TAKASE-HIROKI/currency_rate_forecast")
    
    # トランプ大統領のツイートデータ取得
    gtd.gettwitterdata()
    trump = pd.read_csv('trump.csv', encoding="utf_8")
    
    for time in trump["time"]:
        trump["time"] = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

    # EST1からJST1への変換
    # trump["jst"] = trump["time"] - timedelta(hours = 14)

     # ドル円の１分足データの取得
    ged.getexchangedata(datetime(2019,11,6), datetime(2019,11,7))
   
    rate = pd.read_csv("usd_1min_trump.csv")
    rate = rate[["t", "c"]]
    rate["t"] = pd.to_datetime(rate["t"])

    pd.options.mode.chained_assignment = None

    # 教師データの作成
    rate["target"] = 0
    pos_mask = rate["c"].diff(1) > 0.0
    neg_mask = rate["c"].diff(1) < 0.0
    rate['target'][pos_mask] = 1
    rate['target'][neg_mask] = 0


    masta = pd.merge(trump, rate, left_on='time', right_on='t', how='outer')
    masta = masta[["t", "tweet", "target"]]
    masta.to_csv('masta_data.csv')
    rate.to_csv('rate.csv')


if __name__ == "__main__":
    main()



