# ライブラリのインポート
import plaidml.keras

from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import time
import datetime

def get_tensorflow_data():

    # CSVファイルから過去レートを読み込む
    df = pd.read_csv("usd_1min2019.csv")
    # 終値を1日分移動させる
    df_shift = df.copy()
    df_shift.Close = df_shift.Close.shift(-60)
    
    # 最後の行を除外
    df_shift = df_shift[:-60]
 
    # 念のためデータをdf_2として新しいデータフレームへコピ−
    df_2 = df_shift.copy()
    
    # time（時間）を削除
    del df_2['Date']
    

    # データセットの行数と列数を格納し確認
    n = df_2.shape[0]
    p = df_2.shape[1]
    print("n : ",n)
    print("p : ",p)
    # 訓練データとテストデータへ切り分け
    train_start = 0
    train_end = int(np.floor(0.7*n))
    test_start = train_end + 1
    test_end = n
    data_train = df_2.loc[np.arange(train_start, train_end), :]
    data_test = df_2.loc[np.arange(test_start, test_end), :]
    

    # テストデータの最後2行を表示
    print("data_train",data_train[90:99])
    print(data_train.shape)
    print("data_test",data_test[90:99])
    print(data_test.shape)
    # データの正規化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_train)
    data_train_norm = scaler.transform(data_train)
    data_test_norm = scaler.transform(data_test)
    
    # 特徴量とターゲットへ切り分け
    X_train = data_train_norm[:, 1:]
    y_train = data_train_norm[:, 0]
    X_test = data_test_norm[:, 1:]
    y_test = data_test_norm[:, 0]

    print("x_train : ", X_train)
    print("y_train : ", y_train)
    print("x_test : ", X_test)
    print("y_test : ", y_test)
    # 正規化から通常の値へ戻す
    y_test = y_test.reshape(-1, 1)
    test_inv = np.concatenate((y_test, X_test), axis=1)
    test_inv = scaler.inverse_transform(test_inv)
    print("test_inv : ", test_inv)

    # 訓練データの特徴量の数を取得
    n_stocks = X_train.shape[1]
    print("n_stocks : ",n_stocks)
    # ニューロンの数を設定
    n_neurons_1 = 256
    n_neurons_2 = 128
    print("START SESSION")
    # セッションの開始
    net = tf.compat.v1.InteractiveSession()
    
    # プレースホルダーの作成
    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])
    
    # 初期化
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Hidden weights
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

    # 出力の重み
    W_out = tf.Variable(weight_initializer([n_neurons_2, 1]))
    bias_out = tf.Variable(bias_initializer([1]))

    # 隠れ層の設定（ReLU＝活性化関数）
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    
    # 出力層の設定
    out = tf.transpose(tf.add(tf.matmul(hidden_2, W_out), bias_out))

    # コスト関数
    mse = tf.compat.v1.reduce_mean(tf.math.squared_difference(out, Y))
    
    # 最適化関数
    opt = tf.train.AdamOptimizer().minimize(mse)
    
    # 初期化
    net.run(tf.compat.v1.global_variables_initializer())

    # ニューラルネットワークの設定
    batch_size = 128
    mse_train = []
    mse_test = []
   
    start = time.time()
    
    # 訓練開始！反復処理
    epochs = 200
    for e in tqdm(range(epochs)):
        net.run(opt, feed_dict={X: X_train, Y: y_train})

    # テストデータで予測
    pred_test = net.run(out, feed_dict={X: X_test})
 
    # 予測データの最初の2つを表示
    # pred_test[0][0:2]

    # 予測値をテストデータに戻そう（値も正規化からインバース）
    pred_test = np.concatenate((pred_test.T, X_test), axis=1)
    pred_inv = scaler.inverse_transform(pred_test)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}\n".format(elapsed_time) + "[sec]")
    # テストデータの最後のデータ（正規化前）
    #print(data_test.values[98])
    
    # テストデータの最後のデータ（正規化を戻した後）
    print("test_data:", test_inv)
    
    # モデルが予測したデータ
    print("pred_data", pred_inv)

    # 予測と実際のテストの終値のチャートをプロットしてみよう
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(test_inv[:,1])
    line2, = ax1.plot(pred_inv[:,1])
    plt.savefig('test3.png')
    plt.show()
    
    # MAEの計算
    mae_test = mean_absolute_error(pred_test, pred_inv)
    print("MAE:", mae_test)

if __name__ == '__main__':
    plaidml.keras.install_backend()
    get_tensorflow_data()