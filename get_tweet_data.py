import tweepy
import datetime
import pandas as pd

# ツイートデータ取得全般
def gettwitterdata():
    #Twitter APIを使用するためのConsumerキー、アクセストークン設定
    Consumer_key = 'ytE7cB4Br9R4GxejaYtvds0tb'
    Consumer_secret = 'lwrTLj6pjj8k1rR2q5jod3EEi0h5bpcUswTnl0bdIEJAk1sn4W'
    Access_token = '1183414488143253504-21DwhCeewiDYl0cMw8Ra2UGAc3cWZk'
    Access_secret = 'VnPgbzA0MybT6Qy1r052G87mgfDchcKgfPz9UFafzQF5C'

    #認証
    auth = tweepy.OAuthHandler(Consumer_key, Consumer_secret)
    auth.set_access_token(Access_token, Access_secret)
    api = tweepy.API(auth)

    trump_tweet_list = []
    trump_created_at_list = []
    trump_tweets = api.user_timeline(screen_name="realDonaldTrump", count=500)
    
    for tt in trump_tweets:    
        trump_tweet_list.append(tt.text)
        trump_created_at_list.append(tt.created_at)

    trump_df = pd.DataFrame({'tweet' : trump_tweet_list,'time' : trump_created_at_list})
    trump_df.to_csv('trump.csv', index=False, encoding= 'utf-8')

if __name__ == '__main__':
    gettwitterdata()