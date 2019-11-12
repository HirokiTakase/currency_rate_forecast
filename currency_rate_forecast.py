import pandas as pd 
import os
import numpy as np
import mpl_finance
import matplotlib.pyplot as plt
# import matplotlib.finance as mpf
from mpl_finance import candlestick_ohlc
from datetime import datetime, timedelta
 
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

###########################ローカル関数############################   
# 結果判定
def Disp_Check_Result(message, result):
    if (result == True):
        print("->>>>>| " + message + " -> SUCCESS!：）|->>>>>")
    else :
        print("->>>>>| " + message + " -> FAIL...：(  |->>>>>")

# str型からdatetime1型へ
def str_to_datetime(df):
    for time in df:
        df = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    return df
###################################################################



def main():
    # 実行ファイルの絶対パス取得
    os.chdir("/Users/TAKASE-HIROKI/currency_rate_forecast")

###########################ツイートデータ系###########################  
    
    # トランプ大統領のツイートデータ取得
    check = gtd.gettwitterdata()

    # 結果の表示
    Disp_Check_Result("getting tweet date ", check)
   
    # ツイートデータcsvオープン
    trump = pd.read_csv('trump.csv', encoding="utf_8")
    
    # strからdatetimeへ
    trump["Date"] = str_to_datetime(trump["Date"])

###########################チャートデータ系############################   
    
    # ドル円の１分足データの取得
    check = ged.getexchangedata(datetime(2019,11,11), datetime(2019,11,12))
    # 結果の表示
    Disp_Check_Result("getting exchange-chart ", check)
         
    # チャートデータcsvオープン     
    rate = pd.read_csv("usd_1min_trump.csv")
    rate = rate[["Date", "Close"]]
    rate["Date"] = pd.to_datetime(rate["Date"])

    pd.options.mode.chained_assignment = None


###########################学習系###########################  
    
    # 教師データの作成
    rate["Target"] = 0
    pos_mask = rate["Close"].diff(1) > 0.0
    neg_mask = rate["Close"].diff(1) < 0.0
    rate['Target'][pos_mask] = 1
    rate['Target'][neg_mask] = 0

    masta = pd.merge(trump, rate, left_on='Date', right_on='Date', how='outer')
    masta = masta[["Date", "Tweet", "Target"]]
    masta.to_csv('masta_data.csv')
    rate.to_csv('rate.csv')


if __name__ == "__main__":
    main()



