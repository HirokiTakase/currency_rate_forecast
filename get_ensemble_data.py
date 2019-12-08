# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ohlc
from matplotlib import ticker
import matplotlib.dates as mdates 
import datetime
# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
 
import subprocess

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
# チェイニングのWarningを非表示
pd.options.mode.chained_assignment = None 




def get_ensamble_data():
    subprocess.call("clear")
    
    # 過去レートデータの読み込み
    df = pd.read_csv("./usd_1min2019.csv")
    # データのサイズを確認
    # print(df.shape)

    ################ 特徴量エンジニアリング#############
    # ボリンジャーバンドの算出
    df['Mean'] = df['Close'].rolling(window=20).mean()
    df['Std'] = df['Close'].rolling(window=20).std()
    df['Up'] = df['Mean'] + (df['Std'] * 2)
    df['Down'] = df['Mean'] - (df['Std'] * 2)
    df['Sma150'] = df['Close'].rolling(150).mean()

    # MACDの計算を行う
    df['Ema_12'] = df['Close'].ewm(span=12).mean()
    df['Ema_26'] = df['Close'].ewm(span=26).mean()
    df['Macd'] = df['Ema_12'] - df['Ema_26']
    df['Signal'] = df['Macd'].ewm(span=9).mean()
 
    # nanのrowを削除
    df = df[149:]
    df = df.reset_index(drop=True)

    fig = plt.figure(figsize=(18, 9))
    ax = plt.subplot(1, 1, 1)
    candle_temp = df[:10000]
    candle_temp = candle_temp.reset_index()
    candlestick2_ohlc(
        ax, candle_temp["Open"], candle_temp["High"], candle_temp["Low"],
        candle_temp["Close"], width=0.9, colorup="r", colordown="b"
    )
    
    ax.plot(candle_temp['Sma150'], label = 'Sma150')
    ax.plot(candle_temp['Mean'], label = 'Bollinger Band Mid (20 min)')
    ax.plot(candle_temp['Up'], label = 'Bollinger Band +2σ (20 min)')
    ax.plot(candle_temp['Down'], label = 'Bollinger Band -2σ (20 min)')
   
    plt.savefig('test.png')

    # #macd
    # df.plot()
    # fig2, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
    # ax1.plot(df['Date'], df['Close'])
    # ax2.plot(df['Date'], df['Macd'])
    # ax2.plot(df['Date'], df['Signal'])
    # fig2.tight_layout()
    # ax1.plot(candle_temp['Macd'], label = 'Macd')
    # ax2.plot(candle_temp['Signal'], label = 'Signal')
    # plt.savefig('test_macd.png')

#########################プライスアクションを検出##############
    # データをずらす
    df['Close1'] = df['Close'].shift(1)
    df['Close2'] = df['Close'].shift(2)
    df['High1'] = df['High'].shift(1)
    df['High2'] = df['High'].shift(2)

    # 不要なデータを削除
    df = df[2:]

    # rule 2
    df['Rule2'] = False
    Rule_2_mask = (df['Close1'] - df['Close2']) > 0.0
    df['Rule2'][Rule_2_mask] = True

    # rule 3
    df['Rule3'] = False
    Rule_3_mask = (df['Close'] - df['High1']) > 0.0
    df['Rule3'][Rule_3_mask] = True

    # スラストアップが出現した場所を確認
    # Rule2 = df['Rule2'] == 1.0
    # Rule3 = df['Rule3'] == 1.0
    #print(df[(df['Rule2'] == True) & (df['Rule3'] == True)])

    #df = df[(df['Rule2'] == True) & (df['Rule3'] == True)][0:32]
    
    df_thrustup = df

    # スラストアップが出現した場所を確認
    rule2 = (df_thrustup['Rule2'] == True)
    rule3 = (df_thrustup['Rule3'] == True)
    
    # スラストアップのフラグ追加
    df_thrustup['thrustup'] = False
    df_thrustup['thrustup'][rule2 & rule3] = True
    del df['Rule2']
    del df['Rule3']

    # ターゲットを抽出 = 5分後に+ 1pipで成功（値1）
    df_thrustup['Close_After5min'] = df_thrustup['Close'].shift(-5)
    df_thrustup['Diff'] = df_thrustup['Close_After5min'] - df_thrustup['Close']
    positive = (df_thrustup['Diff'] > 0.01) & (df_thrustup['thrustup'] == True)
    df_thrustup['Target'] = False
    df_thrustup['Target'][positive] = True
    
    # 成功したシグナルの場所を確認
    print(df_thrustup[df_thrustup['Target'] == True])

    ig = plt.figure(figsize=(18, 9))
    ax = plt.subplot(1, 1, 1)
    candle_temp = df[:56851]
    candle_temp = candle_temp.reset_index()
    candlestick2_ohlc(
        ax, candle_temp["Open"], candle_temp["High"], candle_temp["Low"],
        candle_temp["Close"], width=0.9, colorup="r", colordown="b"
    )
    
    ax.plot(candle_temp['Sma150'], label = 'Sma150')
    ax.plot(candle_temp['Mean'], label = 'Bollinger Band Mid (20 min)')
    ax.plot(candle_temp['Up'], label = 'Bollinger Band +2σ (20 min)')
    ax.plot(candle_temp['Down'], label = 'Bollinger Band -2σ (20 min)')
    plt.legend(fontsize = 18)
    plt.savefig('test2.png')

    # データクリーンアップ
    del df_thrustup['Date']
    del df_thrustup['Close1']
    del df_thrustup['Close2']
    del df_thrustup['Close_After5min']
    del df_thrustup['High1']
    del df_thrustup['High2']
    del df_thrustup['thrustup']
    del df_thrustup['Diff']
    
    # 特徴量とターゲットへ分割
    X = df_thrustup[df_thrustup.columns[:-1]].values
    y = df_thrustup['Target'].values
    print(X.shape)
    print(y.shape)

    # 特徴量の標準化
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ####################ロジスティック回帰#######################
    # Sklearnからインポート
    from sklearn.linear_model import LogisticRegression
    
    # モデル訓練
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    # クラスに属する確率を出力（訓練/テストデータ）
    logi_train = clf.predict_proba(X_train)
    logi_test = clf.predict_proba(X_test)
    
    # 結果を出力
    print(logi_test[0:5])

    # ターゲットを閾値で変更
    # 1に属する確率が3%より大きければ「1」、それ以外は「0」
    logi_score = np.where(logi_test[:,1] > 0.03, 1, 0)

    # 混同行列
    matrix = confusion_matrix(y_test, logi_score, labels=[1, 0])
    print(matrix)

    # モデルの評価指標
    print(precision_score(y_test, logi_score))
    print(recall_score(y_test, logi_score))
    print(f1_score(y_test, logi_score))

    ################################# ランダムフォレスト#######################################
    # Sklearnをインポート
    from sklearn.ensemble import RandomForestClassifier
    
    # モデル訓練
    RF = RandomForestClassifier(n_estimators = 512, max_depth=2, class_weight={0:1, 1:50}, random_state=42)
    RF = RF.fit(X_train, y_train)
    
    # 予測出力（訓練/テストデータ）
    rf_train = RF.predict_proba(X_train)
    rf_test = RF.predict_proba(X_test)
    
    # 確認
    print(rf_test[0:5])

    # ターゲットを閾値で変更
    # 1に属する確率が60%より大きければ「1」、それ以外は「0」
    rf_score = np.where(rf_test[:,1] > 0.6, 1, 0)
    
    # 混同行列
    matrix = confusion_matrix(y_test, rf_score, labels=[1, 0])
    print(matrix)

    # 精度（precision）と検出率（Recall）
    print(precision_score(y_test, rf_score))
    print(recall_score(y_test, rf_score))
    print(f1_score(y_test, rf_score))
 
    ############多層パーセンプトロン################
    # ライブラリをインポート
    from sklearn.neural_network import MLPClassifier
    
    # 訓練
    mlp = MLPClassifier(solver="adam", random_state=42, max_iter=500, verbose=True)
    mlp.fit(X_train, y_train)
    
    # 予測（確率）
    mlp_train = mlp.predict_proba(X_train)
    mlp_test = mlp.predict_proba(X_test)
    
    # 確認
    print(mlp_test[0:5])

    # ターゲットを閾値で変更
     # 1に属する確率が2.2%より大きければ「1」、それ以外は「0」
    mlp_score = np.where(mlp_test[:,1] > 0.03, 1, 0)
    
    # 混同行列
    matrix = confusion_matrix(y_test, mlp_score, labels=[1, 0])
    print(matrix)

    # 評価指標を算出
    print(precision_score(y_test, mlp_score))
    print(recall_score(y_test, mlp_score))
    print(f1_score(y_test, mlp_score))

    #subprocess.call('clear')

    # 今までの推測結果データを確認
    print("Train Data------------------")
    print(y_train.shape)
    print(logi_train.shape)
    print(rf_train.shape)
    print(mlp_train.shape)
    print("Test Data------------------")
    print(y_test.shape)
    print(logi_test.shape)
    print(rf_test.shape)
    print(mlp_test.shape)
    print("Check------------------")
    print(y_test[33])
    print(logi_test[33])
    print(rf_test[33])
    print(mlp_test[33])

    ############# XGBoostでスタッキング #################

    # アンサンブル用 訓練データ作成
    X_train_ens = pd.DataFrame(logi_train[:,1], columns=['logi'])
    X_train_ens['rf'] = rf_train[:, 1]
    X_train_ens['mlp'] = mlp_train[:, 1]
    X_train_ens = X_train_ens.values
    y_train_ens = y_train.copy()

    # アンサンブル用 テストデータ作成
    X_test_ens = pd.DataFrame(logi_test[:,1], columns=['logi'])
    X_test_ens['rf'] = rf_test[:, 1]
    X_test_ens['mlp'] = mlp_test[:, 1]
    X_test_ens = X_test_ens.values
    y_test_ens = y_test.copy()

    # XGBoostインポート
    import xgboost as xgb 
    from xgboost import XGBClassifier 
    
    # モデル訓練
    xgboost = xgb.XGBClassifier(random_state=42)
    xgboost.fit(X_train_ens, y_train_ens)
    
    # テストデータで予測
    fin_proba = xgboost.predict_proba(X_test_ens)
    print(fin_proba[0:5])

    # ターゲットを閾値で変更
    # 1に属する確率が3.4%より大きければ「1」、それ以外は「0」
    fin_score = np.where(fin_proba[:,1] > 0.04, 1, 0)

    # 混同行列
    matrix = confusion_matrix(y_test, fin_score, labels=[1, 0])
    print(matrix)

    # F値を確認
    print(precision_score(y_test, fin_score))
    print(recall_score(y_test, fin_score))
    print(f1_score(y_test, fin_score))

    

if __name__ == '__main__':
    get_ensamble_data()