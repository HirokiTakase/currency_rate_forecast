import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import configparser
import pytz
import datetime
from datetime import datetime, timedelta
import plaidml.keras
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tqdm import tqdm
#windowを設定
# window_len = 60
length_of_sequence = 1440
in_out_neurons = 1
hidden_neurons = 300

# PandasのデータフレームからNumpy配列へ変換しましょう
def pd_to_np(data_lstm_in):
  data_lstm_in = [np.array(data_lstm_input) for data_lstm_input in data_lstm_in]  #array のリスト
  data_lstm_in = np.array(data_lstm_in) #np.array
  return data_lstm_in

# LSTMへの入力用に処理の関数
# 60*24=1440
def data_maker(data, lenght_of_sequence):
  data_lstm_in=[]
  docX , docY = [], []

  if len(data)==lenght_of_sequence:
    print('入りました')
    temp = data[:lenght_of_sequence].copy()
    temp = temp / temp.iloc[0] - 1
    data_lstm_in.append(temp)
    for i in range(len(data) - lenght_of_sequence):
      # temp = data[i:(i + lenght_of_sequence)].copy()
      # temp = temp / temp.iloc[0] - 1
      # data_lstm_in.append(temp)
      docX.append(data.iloc[i:(i + lenght_of_sequence)].as_matrix())
      docY.append(data.iloc[i + lenght_of_sequence].as_matrix())
  alsX = np.array(docX)
  alsY = np.array(docY)
  return alsX, alsY

def data_maker2(data):
  print('datamaker2')
  data_lstm_in=[]
  if len(data)==60:
    temp = data[:60].copy()
    temp = temp / temp.iloc[0] - 1
    data_lstm_in.append(temp)
  leng = len(data) - 60
  for i in tqdm(range(leng)):
      temp = data[i:(i + 60)].copy()
      temp = temp / temp.iloc[0] - 1
      data_lstm_in.append(temp)
  return data_lstm_in


# LSTMのモデルを設定
def build_model(neurons, lenght_of_sequence, in_out_neurons):
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(None, lenght_of_sequence, in_out_neurons),return_sequences = False))
    model.add(Dense(in_out_neurons))
    model.add(Activation('linear'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model


def build_model2(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
 
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
 
    model.compile(loss=loss, optimizer=optimizer)
    return model
def main():

  res = pd.read_csv('usd_1min2019.csv')
  df = res[['Date', 'Open']]
  
  # df["Date"] = pd.to_datetime(df["Date"])

  print(df['Date'].dtype)
  split_date = '2019-11-21 07:56:00'
  train, test = df[df['Date'] < split_date], df[df['Date']>=split_date]
  # del train['Date']
  # del test['Date']
  latest = test[:60]
  del train['Date']
  del test['Date']
  del latest['Date']
  length = len(test)- 60
  print('df',df)

  # scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
  # train = scaler.fit_transform(train)
  # test = scaler.fit_transform(test)

  print('train', train)
  print('test',test)


  # ランダムシードの設定
  np.random.seed(seed=202)
  # X_train, Y_train = data_maker(train, length_of_sequence)
  # X_test, Y_test = data_maker(test, length_of_sequence)

  train_lstm_in = data_maker2(train)
  lstm_train_out = (train['Open'][60:].values / train['Open'][:-60].values)-1
  test_lstm_in = data_maker2(test)
  lstm_test_out = (test['Open'][60:].values / test['Open'][:-60].values)-1
  latest_lstm_in = data_maker2(latest)
  # print(X_test.shape[1])
  train_lstm_in = pd_to_np(train_lstm_in)
  test_lstm_in = pd_to_np(test_lstm_in)
  latest_lstm_in = pd_to_np(latest_lstm_in)
  # X_train, Y_train = data_maker(train, length_of_sequence)
  # X_test, Y_test = data_maker(test, length_of_sequence)
  # print(X_test.shape[1])
  print('モデル作成中')
  # 初期モデルの構築
  yen_model = build_model2(train_lstm_in, output_size=1, neurons = 20)
  print('モデル作成終了')
  print('フィッティング開始')
  # データを流してフィッティングさせましょう
  yen_history = yen_model.fit(train_lstm_in, lstm_train_out, 
                            epochs=5, batch_size=256, verbose=2, )
  print('フィッティング終了')
  # X_train = np.reshape(X_train, (X_train.shape[0], 3, X_train.shape[1]))
  # X_test = np.reshape(X_test, (X_test.shape[0], 3, X_test.shape[1]))
  # 初期モデルの構築
  # yen_model = build_model(hidden_neurons, length_of_sequence, in_out_neurons)
  
  # データを流してフィッティングさせましょう
  # yen_history = yen_model.fit(X_train, Y_train, batch_size = 100, nb_epoch = 100 ,validation_split = 0.05)
  # predicted = yen_history(X_test)
  # result = pd.DataFrame(predicted)
  # result.comumns = ['predicted']
  # result['actual'] = Y_test
  # result.plot()
  # plt.show

  print(yen_history)


  # #未来の値
  empty = []
  future_array = np.array(empty)
  for i in range(length):
    pred = (((np.transpose(yen_model.predict(latest_lstm_in))+1) * latest['Open'].values[0])[0])[0]
    future_array= np.append(future_array,pred)
    data ={'Open':[pred]}
    df1 = pd.DataFrame(data)
    latest =pd.concat([latest,df1],axis=0)
    latest.index = range(0,60+1)
    latest = latest.drop(0,axis=0)
    latest_lstm_in =pd_to_np(latest_lstm_in)
    
    print(df1)
    print(latest_lstm_ins)
    print(latest)



  plt.figure(figsize=(10,8))
  plt.plot(df[df['Date']< split_date]['Date'][60:],
         train['Open'][60:], label='Actual', color='blue')
  plt.plot(df[df['Date']< split_date]['Date'][60:],
         ((np.transpose(yen_model.predict(train_lstm_in))+1) * train['Open'].values[:-60])[0], 
         label='Predicted', color='red')
  plt.show()
  plt.savefig('tre.png')

  plt.figure(figsize=(10,8))
  plt.plot(df[df['Date']>= split_date]['Date'][60:],
         test['Open'][60:], label='Actual', color='blue')
  plt.plot(df[df['Date']>= split_date]['Date'][60:],
        future_array,label='future',color='green')
  plt.show()
  plt.savefig('tre1.png')
if __name__ == '__main__':
  print("shit")
  plaidml.keras.install_backend()
  main()