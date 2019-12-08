import tweepy
import datetime
import pandas as pd
import time
from tqdm import tqdm
# ツイートデータ取得全般
def gettwitterdata(start_day, end_day):
    #Twitter APIを使用するためのConsumerキー、アクセストークン設定
    Consumer_key = 'ytE7cB4Br9R4GxejaYtvds0tb'
    Consumer_secret = 'lwrTLj6pjj8k1rR2q5jod3EEi0h5bpcUswTnl0bdIEJAk1sn4W'
    Access_token = '1183414488143253504-21DwhCeewiDYl0cMw8Ra2UGAc3cWZk'
    Access_secret = 'VnPgbzA0MybT6Qy1r052G87mgfDchcKgfPz9UFafzQF5C'

    #認証
    auth = tweepy.OAuthHandler(Consumer_key, Consumer_secret)
    auth.set_access_token(Access_token, Access_secret)
    api = tweepy.API(auth)

    df_old = pd.read_csv('./trump.csv')
    df_old['Date'] = pd.to_datetime(df_old['Date'])
    start_time =  pd.to_datetime(df_old["Date"][0])+ datetime.timedelta(seconds = 1) - datetime.timedelta(hours = 9)
    end_time = datetime.datetime.now() - datetime.timedelta(hours = 9)

    # 取得データを表示
    print("SEARCH START TIME : ", start_time + datetime.timedelta(hours = 9))
    print("SEARCH END TIME   : ", end_time +  datetime.timedelta(hours = 9), '\n')
    # リストの初期化
    trump_tweet_list = []
    trump_created_at_list = [] 
    
    # 100 * 10 ツイート取ってきているけどそんなにいらないかも
    for i in tqdm(range(7000)):
        # APIでツイート取得
        trump_tweets = api.user_timeline(screen_name="realDonaldTrump", 
                                        count=100, page = i)
        # 対象の期間のツイートデータをリストに追加
        for tt in trump_tweets:    
            tt.created_at = tt.created_at + datetime.timedelta(hours = 9)
            if start_time <= tt.created_at and tt.created_at <= end_time:
                trump_tweet_list.append(tt.text)
                trump_created_at_list.append(tt.created_at)
    print('\n')
    
    # リストをdfに変換      
    trump_df = pd.DataFrame({'Tweet' : trump_tweet_list,'Date' : trump_created_at_list})
    trump_df['Date'] = pd.to_datetime(trump_df["Date"])
    # 秒を落とすs
    trump_df["Date"] = trump_df["Date"].dt.floor('Min')
    # 過去データとマージ
    trump_df = pd.concat([trump_df, df_old])
  

   
    # データ保存
    trump_df.to_csv('./trump.csv', index=False, encoding= 'utf-8')
    
    return True 

if __name__ == '__main__':
    gettwitterdata(datetime.datetime(2019,11,10), datetime.datetime(2019,11,11))