# データ処理のライブラリ
import pandas as pd
import numpy as np 
import datetime
 
# Matplotlibのインポート
import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ohlc
from matplotlib import ticker
import matplotlib.dates as mdates 

# Talibのインポート
import talib as ta

# ローソク足チャート
def candlechart(data,save_name):
    width=0.8
    fig, ax = plt.subplots()
 
    # ローソク足
    candlestick2_ohlc(ax, opens=data.Open.values, closes=data.Close.values,
                          lows=data.Low.values, highs=data.High.values,
                          width=width, colorup='r', colordown='b')
 
    xdate = data.index
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
 
    def mydate(x, pos):
        try:
            return xdate[int(x)]
        except IndexError:
            return ''
 
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))
    ax.format_xdata = mdates.DateFormatter('%m-%d')
 
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.savefig(save_name)
    return fig, ax
 
def get_pattern_matching_data():

    # CSVファイルの読み込み
    masta = pd.read_csv('usd_1min_trump.csv')
    df = masta.copy()

    # データの日付をIndexとして使う
    df.index = pd.to_datetime(df['Date'])
    
    # 不要なカラムの削除
    del df['Date']
    # del df['Unnamed: 0']

    # OHLCデータをNumpy配列へ変換
    Open = np.array(df['Open'])
    Close = np.array(df['Close'])
    Low = np.array(df['Low'])
    High = np.array(df['High'])

    #####################################################################################################    
    # ローソク足パターンを抽出
    df['Marubozu'] = ta.CDLMARUBOZU(Open, High, Low, Close) # 丸坊主
    df['Hammer'] = ta.CDLHAMMER(Open, High, Low, Close) # ハンマー
    df['Inverted_Hammer'] = ta.CDLINVERTEDHAMMER(Open, High, Low, Close) # 逆ハンマー
    df['Short_Line_Candle_Up-Down'] = ta.CDLSHORTLINE(Open, High, Low, Close) # 短い足のローソク
    df['Long_Line_Candle_Up-Down'] = ta.CDLLONGLINE(Open, High, Low, Close) # 長い足のローソク
    
    # 同時線（とんぼ）
    df['Dragonfly_Doji'] = ta.CDLDRAGONFLYDOJI(Open, High, Low, Close)
    
    # つつみ線
    df['Engulfing_Pattern'] = ta.CDLENGULFING(Open, High, Low, Close)

    # パターン
    df['Two_Crows'] = ta.CDL2CROWS(Open, High, Low, Close) # カブセ線
    df['Three_Black_Crows'] = ta.CDL3BLACKCROWS(Open, High, Low, Close) # 三羽鳥
    df['Three_Outside_Up-Down'] = ta.CDL3OUTSIDE(Open, High, Low, Close) # 抱き線
    #######################################################
    # 【買いのサイン】１本目の陰線は下ヒゲが長く、実体も長い。また、二番目のロウソクは短く、黒の丸坊主、星で二番目のセッションに関連する                               
    # 【売りのサイン】 高い終値の3つの白いろうそく足最初の2つのろうそく足 は長くなっています。それぞれのろうそく足の始値は前の実体にかぶさっています。
    df['Three_Stars_In_The_South'] = ta.CDL3STARSINSOUTH(Open, High, Low, Close)

    #【買いのサイン】3つの長い白のろうそく足が次々と現れ、それぞれの終値は前の終値より高くなっています。
    df['Three_Advancing_White_Soldiers'] = ta.CDL3WHITESOLDIERS(Open, High, Low, Close) 
    
    # 【買いのサイン】
    # 一日目に陰線が出現した翌日、それより下に十字星、極性が現れる。その後、三本目のロウソク足はヒゲの間に同じ隙間のある「長い」陽線のろうそく足で、一番目より短くなっています。
    # 【売りのサイン】
    # 一日目に陽線が出現した翌日、それより上に十字星、極性が現れる。その後、三本目のロウソク足はヒゲの間に同じ隙間のある「長い」陰線のろうそく足で、一番目より短くなっています。
    df['Abandoned_Baby'] = ta.CDLABANDONEDBABY(Open, High, Low, Close) 

    # 【買いのサイン】
    # 上昇トレンド内、もしくは下落トレンド内での反発を意味している。
    # また、陽線の実体はだんだんと短くなっていく。2番目と3番目のキャンドルの開きは、前のキャンドルの実体の中にあります。3つのキャンドルの上のヒゲは次第に高くなります  
    df['Advance_Block'] = ta.CDLADVANCEBLOCK(Open, High, Low, Close)
    
    # 【買いのサイン】
    # レンド方向に大差があるろうそく足のオープン
    # 白ろうそく足（陽の大引け坊主）
    # 白いろうそく足の実体は前のろうそく足の実体よりもずっと大きくなっています。
    # 【売りのサイン】
    # レンド方向に大差があるろうそく足のオープン
    # 黒ろうそく足 （陰の大引け坊主）
    # 黒いろうそく足の実体は前のろうそく足の実体よりもずっと大きくなっています。
    df['Belt_hold'] = ta. CDLBELTHOLD(Open, High, Low, Close)
    
    # 【買いのサイン】
    # 初日の大きい陰線は今の下落トレンド象徴して、次の日もまた陰線はトレンドの方向に実体の間隔がある。
    # 【売りのサイン】
    # 初日の大きい陽線は今の下落トレンド象徴して、次の日もまた陽線はトレンドの方向に実体の間隔がある。
    df['Breakaway'] = ta.CDLBREAKAWAY(Open, High, Low, Close)
    
    # 【買いのサイン】
    # 最初の二本の陰線は丸坊主。
    # 三番目は二番目のろうそく足長さ範囲内で形作られます。それは長い上ヒゲを形成します。
    # 四本目は陰線となり、抱き込む形になる。
    df['Concealing_Baby_Swallow'] = ta.CDLCONCEALBABYSWALL(Open, High, Low, Close)
   
    # 【買いのサイン】
    # 最初は下落トレンドであり、最初のローソク足は陰線で実体が長い。
    # 二本目のローソクは一本目のローソクと同じような大きさで終値が一本目のローソクの終値と近い陽線。今後上昇トレンドに変化する事がある。
    # 【売りのサイン】
    # 最初は上昇トレンドであり、最初のローソク足は陽線で実体が長い。
    # 二本目のローソクは一本目のローソクと同じような大きさで終値が一本目のローソクの終値と近い陰線。今後下落トレンドに変化する事がある。
    df['Counterattack'] = ta.CDLCOUNTERATTACK(Open, High, Low, Close)
   
    # 【売りのサイン】
    # どちらのろうそく足も長くなっています。
    # 黒の ろうそく足の始値は白のろうそく足の高値を上回ります。
    # ※ブルベアの三年間で１〜２回反応する程度
    df['Dark_Cloud_Cover'] = ta.CDLDARKCLOUDCOVER(Open, High, Low, Close)
    
    # Doji（同時線）とは？
    # ろうそく足の実体が小さいため、始値と終値が同じであればそれは「同時線」と呼ばれます。
    # 始値と終値の要求が全く同じであり、データに厳しい制約が設けられるので、同時線はほとんど見られません。
    # 始値と終値の差が数個のティック（最小価格変動）を超えることがなければ、これは十分以上です。
    df['Doji'] = ta.CDLDOJI(Open, High, Low, Close)
    
    # 【買いのサイン】
    # 最初は長い陰線となり、二本目はトレンド方向のブレイクがある同時線が出る時。
    # image.png
    # 【売りのサイン】
    # 最初は長い陽線となり、二本目はトレンド方向のブレイクがある同時線が出る時。
    df['Doji_Star'] = ta.CDLDOJISTAR(Open, High, Low, Close)
    # 説明なし
    df['Evening_Doji_Star'] = ta.CDLEVENINGDOJISTAR(Open, High, Low, Close)
    
    # 【買いのサイン】
    # 一番目と三番目のセッションは「長い」ろうそく足です。星のヒゲは短く、色は関係ありません。最初のろうそく足の終値から星は離れています。
    # 三番目のろうそく足は一番目より短くその長さの内側に収まっています。
    # 【売りのサイン】
    # 一番目と三番目のセッションは「長い」ローソク足です。星のヒゲは短く、色は関係ありません。最初のろうそく足の終値から星は離れています。
    # 三番目のろうそく足は一番目より短くその範囲に収まっています。
    df['Evening_Star'] = ta.CDLEVENINGSTAR(Open, High, Low, Close)
   
    # この長い足がある同時線は強い上昇トレンドや強い下落トレンドで最も重要とされている。
    # 理由は足の長い同時線は、需要と供給の力が均衡に近づいており、トレンドの逆転が起こる可能性があることを示唆しています。
    df['Long Legged Doji'] = ta.CDLLONGLEGGEDDOJI(Open, High, Low, Close)
    
    # 【買いのサイン】
    # 最初に陰線の丸坊主が出てきてその後に陽線の丸坊主が下落トレンドで出てくる。
    # その時最初のローソクと次のローソクにはギャップアップがあり、ローソクが長ければ逆（今回の場合上昇トレンド）になる信憑性が高くなる。
    # 【売りのサイン】
    # 最初に陽線の丸坊主が出てきてその後に陰線の丸坊主が上昇トレンドで出てくる。
    # その時最初のローソクと次のローソクにはギャップダウンがあり、ローソクが長ければ逆（今回の場合下落トレンド）になる信憑性が高くなる。
    df['Kicking'] = ta.CDLKICKING(Open, High, Low, Close)
    
    # 【買いのサイン】
    # このパターンは上昇トレンドに反転する時のパターンの一つであり下落トレンド時にでる。
    # 一本、二本、三本目そして四本目のローソクは陰線で長い実体、それぞれの初値と終値は前のローソクの初値と終値の間にある。
    # 四本目の陰線は実体が短く上ヒゲがある。
    # 五本目は前のローソク足の実体を始値が抜いている陽線。
    df['Ladder_Bottom'] = ta.CDLLADDERBOTTOM(Open, High, Low, Close)
    
    # [買いのサイン]
    # 下落トレンドであり、最初のローソク足は実体が長い陰線であり、二本目のローソクは最初のローソク足の終値がほぼ同じ終値の陰線。2番目のろうそくが1番目のろうそくの終値を下回ることができないということは、強気の逆転に対する支持レベルを生み出します。このパターンは、数期間にわたって価格が下落傾向にある時間帯ではなく、大規模な上昇トレンドに続く一時的な下落に最も適しています。
    # ※[売りのサイン]にもなりうる
    df['Matching_Low'] = ta.CDLMATCHINGLOW(Open, High, Low, Close)
    
    # [買いのサイン]
    # 強気キッカーローソク足パターンを特定するには、以下の基準を探します。
    # まず、最初のキャンドルは黒または弱気のローソク足である必要があります。次に、2番目のキャンドル（白または強気）が最初のキャンドルの終点より上に開いている必要があります。第三に、第二のろうそく足の形成中の価格の動きは、第一と第二のろうそくの間に形成されたギャップに決して落ちないはずです。ご想像のとおり、これは2番目のローソク足に一番下の芯があることはめったにないことを意味します。
    # Bullish Kickerローソク足パターンは、大幅な下落の後に形成する必要はありませんが、多くの場合そうなります。これが起こるとき、態度の突然の変化はたぶんゲームを変えるニュースイベントが原因です。
    # 最後に、もしあなたが反対の形成（白いろうそくとそれに続く隙間に落ちない黒いろうそく）でパターンを見つけるなら、あなたはあなたの手に弱気キッカーがあるかもしれません。それはさらに下向きの動きを予告する弱気な反転信号です。強気のキッカーのように、弱気のキッカーはまれですが信頼できることを証明します
    df['Mat_Hold'] = ta.CDLMATHOLD(Open, High, Low, Close)
    
    # 【売りのサイン】
    # 上昇トレンドに見れる。上ヒゲは3以下ではなく実体より大きい。
    # 下ヒゲはない、または非常に短い。（ろうそく足範囲の10%以上ではない。）
    # 星と以前のろうそく足間の価格差
    df['Shooting_Star'] = ta.CDLSHOOTINGSTAR(Open, High, Low, Close)
    
    df.to_csv("pattern_matching_result.csv")
    #####################################################################################################

    # データの確認
    print(df[10:15])

    #丸坊主のデータを確認
    # Ta-Libのローソク足パターン分析では、「-100」「0」「100」の値が戻ってきます。
    # 0はパターンが出ていない状態、
    # 100は陽線のパターンが検出された場合、-
    # 100は陰線のパターンが検出したことを表します。
    df[(df['Marubozu'] < 0) | (df['Marubozu'] > 0)].head()
    
    # 丸坊主のカウントを確認
    print(df['Marubozu'][(df['Marubozu'] < 0) | (df['Marubozu'] > 0)].count())

    # 丸坊主のデータ確認
    print(df['Marubozu'].loc['2019-07-15 09:14:00'])

    # 丸坊主 陽線（上昇トレンドが力強い）
    set_time = datetime.datetime.strptime('2019-07-15 09:14:00', '%Y-%m-%d %H:%M:%S')
    before = set_time - datetime.timedelta(minutes=10)
    after = set_time + datetime.timedelta(minutes=10)
    candlechart(df.loc[(df.index > before )&(df.index < after)],'marubouzu_up.png')

    # 陰線丸坊主が検出されたレコード
    print(df['Marubozu'].loc['2019-07-15 9:33:00'])

    # 丸坊主 陰線（下降トレンドが力強い）
    set_time = datetime.datetime.strptime('2019-07-15 9:33:00', '%Y-%m-%d %H:%M:%S')
    before = set_time - datetime.timedelta(minutes=10)
    after = set_time + datetime.timedelta(minutes=10)
    candlechart(df.loc[(df.index > before )&(df.index < after)], 'marubouzu_down.png')
 




    # 陽線包み線（値100）、陰線包み線（値-100）をカウントしてみる
    print(df['Engulfing_Pattern'][(df['Engulfing_Pattern'] < 0) | (df['Engulfing_Pattern'] > 0)].count())


    # 陽線 包み線 下降トレンドから上昇トレンドへ
    set_time = datetime.datetime.strptime('2019-07-15 10:44:00', '%Y-%m-%d %H:%M:%S')
    before = set_time - datetime.timedelta(minutes=10)
    after = set_time + datetime.timedelta(minutes=10)
    candlechart(df.loc[(df.index > before )&(df.index < after)], 'yousen_fukumi.png')

    # 陰線の包み線 上昇トレンドから下降トレンドへ
    set_time = datetime.datetime.strptime('2019-07-15 10:49:00', '%Y-%m-%d %H:%M:%S')
    before = set_time - datetime.timedelta(minutes=10)
    after = set_time + datetime.timedelta(minutes=10)
    candlechart(df.loc[(df.index > before )&(df.index < after)], 'insen_fukumi.png')


    # 陰線 カラカサ （高値圏から下降トレンドへ転換）
    set_time = datetime.datetime.strptime('2019-11-05 1:21:00', '%Y-%m-%d %H:%M:%S')
    before = set_time - datetime.timedelta(minutes=10)
    after = set_time + datetime.timedelta(minutes=10)
    candlechart(df.loc[(df.index > before )&(df.index < after)], 'yousen_karakasa.png')

    # トンボ 
    set_time = datetime.datetime.strptime('2019-11-05 4:30:00', '%Y-%m-%d %H:%M:%S')
    before = set_time - datetime.timedelta(minutes=10)
    after = set_time + datetime.timedelta(minutes=10)
    candlechart(df.loc[(df.index > before )&(df.index < after)],'tonbo.png')


if __name__ == '__main__':
    get_pattern_matching_data()