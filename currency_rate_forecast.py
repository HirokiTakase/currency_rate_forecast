import pandas as pd 
import os
import numpy as np
import mpl_finance
import matplotlib.pyplot as plt
# import matplotlib.finance as mpf
from mpl_finance import candlestick2_ohlc
from datetime import datetime, timedelta
 
# # 自然言語処理（NLP）を扱うライブラリ
import nltk
from nltk.corpus import stopwords
import string
 

import time
from tqdm import tqdm
import subprocess
import sys
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

# start_day = datetime(2011,2,10,0,0,0)
# end_day = datetime(2012,2,13,23,59,00)

###########################ローカル関数############################   
# 結果判定
def Disp_Check_Result(message, result):
    if (result == True):
        print("                        ----------------------------------------------")
        print("                        | ->>>>>| " + message + "-> [SUCCESS!]：）|")  
        print("                        ----------------------------------------------")
        print("\n")
    else :
        print("                        ----------------------------------------------")
        print("                        | ->>>>>| " + message + "-> [FAIL...]：(  |")
        print("                        ----------------------------------------------")
        print("\n")

# 言語解析用
def tweet_clean(input):
    clean = ''.join([char for char in input if char not in string.punctuation]).lower()
    clean = ''.join([char for char in clean if char not in "“”’" ])
    clean = ' '.join([word for word in clean.split() if word not in (stopwords.words('english'))])
    return clean

###################################################################


###################################################################
def main():
    
    subprocess.call('clear')
    # 実行ファイルの絶対パス取得
    os.chdir("/Users/TAKASE-HIROKI/currency_rate_forecast/currency_rate_forecast/")

###########################ツイートデータ系###########################  
    
    # for i in tqdm(range(1)):
    #     # トランプ大統領のツイートデータ取得
    #     check = gtd.gettwitterdata(start_day, end_day)
    #     time.sleep(2)
    
    # # 結果の表示
    # Disp_Check_Result("getting tweet date ", check)
    # subprocess.call('clear')
    # # ツイートデータcsvオープン
    # trump = pd.read_csv('trump.csv', encoding="utf_8")
    
    # # strからdatetimeへ
    # trump['Date'] = pd.to_datetime(trump['Date'], format='%Y/%m/%d %H:%M:%S')

###########################チャートデータ系############################   
    
    for i in tqdm(range(25)):
        # ドル円の１分足データの取得
        try:
            check = ged.getexchangedata(start_day, end_day)
            # time.sleep(0.01)
        except OverflowError as e:
            print(e)
            break 
        subprocess.call('clear')
    
    time.sleep(1)    
    sys.exit() #トル円データのみを取得用
    # 結果の表示
    Disp_Check_Result("getting exchange chart ", check)
    time.sleep(1)   
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

    masta = pd.merge(trump, rate, on = "Date", how = 'inner')
    # masta = masta[["Date", "Tweet", "Target"]]
   
    masta.to_csv('masta_data.csv')
    rate.to_csv('rate.csv')

    masta = pd.read_csv("masta_data.csv",index_col = 0)
    # print(masta)
    pd.options.mode.chained_assignment = None
    # データへ適用
    masta['Tweet'] = masta['Tweet'].apply(tweet_clean)
    del masta['Date']

    # テストデータと訓練データへ分割
    train, test = train_test_split(masta, test_size=0.2, random_state=1)
    # CountVectroizer()のインスタンス
    vect = CountVectorizer()
    # 訓練データを学習してBag of wordsへ変換
    X_train = vect.fit_transform(train['Tweet'])
    # テストデータは変換のみ（訓練はしない)
    X_test = vect.transform(test['Tweet'])
    # 訓練データのターゲット
    y_train = train['Target'].values
    # テストデータのターゲット
    y_test = test['Target'].values
    # 多項分布・ナイーブベイズのモデルを訓練
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    # 訓練データから予測してみる
    train_pred = clf.predict(X_train)
    # 混同行列を表示
    print(confusion_matrix(y_train, train_pred))
   
    # 正解率
    print(accuracy_score(y_train, train_pred))
    # テストデータで予測
    test_pred = clf.predict(X_test)
    print(test_pred)
    # 混同行列を表示
    print(confusion_matrix(y_test, test_pred))
    # 正解率
    print(accuracy_score(y_test, test_pred))
    print(test_pred[0])
    print("=====================")
    print(test.iloc[0])
    # 該当ツイートの時間を確認
    trump = pd.read_csv('trump.csv')
    trump.query('Tweet.str.contains("300th")', engine='python')
    # 為替レートデータを再度読み込み
    rate_data = pd.read_csv('usd_1min_trump.csv')
    rate_data[rate_data['Date'] == '2019-11-11 06:12:00']
    # ローソク足表示
    fig = plt.figure(figsize=(18, 9))
    ax = plt.subplot(1, 1, 1)
    candle_temp = rate_data[4:100]
    candle_temp = candle_temp.reset_index()
    candlestick2_ohlc(
    ax, candle_temp["Open"], candle_temp["High"], candle_temp["Low"],
    candle_temp["Close"], width=0.9, colorup="r", colordown="b")
    plt.show

if __name__ == "__main__":
    main()





