# Tokyo house price prediction

2020/05/16  
Yosuke Kobayashi  
m7142yosuke@gmail.com

---
## このnotebookについて
データ分析を目的として保存されていないデータは非常に汚く、予測や可視化をおこなうには前処理が必要です。  
このnotebookでは東京の住宅価格予測を例にスクレイピングで取得した汚い生のデータを加工し、可視化・予測・精度評価するまでの手順の一例を示します。  
## 工夫したこと
## 分析して得られた知見
## 予測精度について
## データの取得方法
2019年末にスクレイピングによりHOME'Sから取得しました。  
約16万5千軒の情報があります。

# データの前処理とEDA
---
## The DATA


```python
model_save = False
```


```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
import warnings
warnings.simplefilter('ignore')
sns.set()

p = Path('../data')
df_all = pd.concat([pd.read_csv(f, index_col=0) for f in p.glob('*.csv')], sort=True)
```


```python
df_all.drop_duplicates(subset='url', keep='first', inplace=True)
```


```python
df_all.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 164716 entries, 0 to 34
    Data columns (total 47 columns):
     #   Column                     Non-Null Count   Dtype 
    ---  ------                     --------------   ----- 
     0   address                    164685 non-null  object
     1   air_conditioner            164716 non-null  int64 
     2   auto_lock                  164716 non-null  int64 
     3   balcony_space              164685 non-null  object
     4   bank                       159775 non-null  object
     5   conditions                 164684 non-null  object
     6   contract_period            139187 non-null  object
     7   daylight_direction         164685 non-null  object
     8   deposit                    164685 non-null  object
     9   deposit2                   164685 non-null  object
     10  eatting                    159775 non-null  object
     11  education                  159775 non-null  object
     12  equipments                 162071 non-null  object
     13  equipments2                142073 non-null  object
     14  floor                      104790 non-null  object
     15  floor2                     164684 non-null  object
     16  floor_plan                 164685 non-null  object
     17  floor_space                164685 non-null  object
     18  flooring                   164716 non-null  int64 
     19  home_insurance             164685 non-null  object
     20  hospital                   159775 non-null  object
     21  how_old                    164685 non-null  object
     22  kichen                     164475 non-null  object
     23  location                   164684 non-null  object
     24  more_than_2                164716 non-null  int64 
     25  name                       164685 non-null  object
     26  new_house                  164716 non-null  int64 
     27  number_of_stations_10_min  160115 non-null  object
     28  number_of_stations_all     160115 non-null  object
     29  other                      148259 non-null  object
     30  parking                    164716 non-null  int64 
     31  pets                       164716 non-null  int64 
     32  public_facility            159760 non-null  object
     33  recomend_point             155964 non-null  object
     34  reheating                  164716 non-null  int64 
     35  renewal_fee                139187 non-null  object
     36  rent                       164685 non-null  object
     37  rent_all                   164685 non-null  object
     38  security_deposit           164685 non-null  object
     39  separate                   164716 non-null  int64 
     40  shopping                   159775 non-null  object
     41  south                      164716 non-null  int64 
     42  station_express_info       164676 non-null  object
     43  status                     164685 non-null  object
     44  structure                  164684 non-null  object
     45  traffic                    164685 non-null  object
     46  url                        164716 non-null  object
    dtypes: int64(10), object(37)
    memory usage: 60.3+ MB



```python
pd.set_option('display.max_columns', 100)
```


```python
df_all.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
      <th>air_conditioner</th>
      <th>auto_lock</th>
      <th>balcony_space</th>
      <th>bank</th>
      <th>conditions</th>
      <th>contract_period</th>
      <th>daylight_direction</th>
      <th>deposit</th>
      <th>deposit2</th>
      <th>eatting</th>
      <th>education</th>
      <th>equipments</th>
      <th>equipments2</th>
      <th>floor</th>
      <th>floor2</th>
      <th>floor_plan</th>
      <th>floor_space</th>
      <th>flooring</th>
      <th>home_insurance</th>
      <th>hospital</th>
      <th>how_old</th>
      <th>kichen</th>
      <th>location</th>
      <th>more_than_2</th>
      <th>name</th>
      <th>new_house</th>
      <th>number_of_stations_10_min</th>
      <th>number_of_stations_all</th>
      <th>other</th>
      <th>parking</th>
      <th>pets</th>
      <th>public_facility</th>
      <th>recomend_point</th>
      <th>reheating</th>
      <th>renewal_fee</th>
      <th>rent</th>
      <th>rent_all</th>
      <th>security_deposit</th>
      <th>separate</th>
      <th>shopping</th>
      <th>south</th>
      <th>station_express_info</th>
      <th>status</th>
      <th>structure</th>
      <th>traffic</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>東京都中野区新井5丁目</td>
      <td>1</td>
      <td>1</td>
      <td>-</td>
      <td>\n（株）りそな銀行 中野支店 新井薬師出張所\n中野上高田郵便局\n西武信用金庫 薬師駅前...</td>
      <td>\n                                            ...</td>
      <td>2年間</td>
      <td>南西</td>
      <td>- / -</td>
      <td>\n                -</td>
      <td>\n（株）奄美ふくらしゃ\nかざみどり\n竜苑\n季の葩\n豊年屋\n</td>
      <td>\nあけぼの保育園\n新井小学校\n大妻中野高等学校\n東亜学園\n目白学園 目白大学 研究...</td>
      <td>\n                    オートロック\n                ...</td>
      <td>\n                    クローゼット\n                ...</td>
      <td>4階/401</td>
      <td>4階 / 4階建\n</td>
      <td>\n    ワンルーム\n     ( 洋室 5.8帖(4階)\n1R )</td>
      <td>19.28m²</td>
      <td>1</td>
      <td>要</td>
      <td>\nクリニックヨコヤマ\n大竹歯科医院\n新渡戸記念中野総合病院\n寺内医院\n総合東京病院\n</td>
      <td>2019年7月 ( 新築 )</td>
      <td>\n                    IHコンロ\n                 ...</td>
      <td>\n                                            ...</td>
      <td>1</td>
      <td>FARE中野7</td>
      <td>1</td>
      <td>徒歩10分以内（1駅）</td>
      <td>すべての駅（2駅）</td>
      <td>\n                                    保証会社要加入保...</td>
      <td>0</td>
      <td>0</td>
      <td>\n中野区役所 公園事務所哲学堂公園事務所\n中野区立 上高田図書館\n豊島区役所 区民ひろ...</td>
      <td>RC造新築デザイナーズ1Rマンション</td>
      <td>1</td>
      <td>87,000円</td>
      <td>8.7万円</td>
      <td>8.7万円 ( 5,000円 )</td>
      <td>8.7万円 / 無</td>
      <td>1</td>
      <td>\nローソンストア１００中野新井四丁目店\nｍｉｎｉピアゴ新井５丁目店\nあらいやくし薬局\...</td>
      <td>0</td>
      <td>西武新宿線 新井薬師前駅 徒歩4分普通準急急行通勤急行ＪＲ中央線 中野駅 徒歩18分普通快速...</td>
      <td>空家</td>
      <td>\n                                            ...</td>
      <td>\n西武新宿線 新井薬師前駅 徒歩4分\nＪＲ中央線 中野駅 徒歩18分\n\n通勤・通学駅...</td>
      <td>https://www.homes.co.jp/chintai/b-1064980018577/</td>
    </tr>
    <tr>
      <th>1</th>
      <td>東京都墨田区東向島3丁目</td>
      <td>1</td>
      <td>1</td>
      <td>-</td>
      <td>\n東京東信用金庫 本店\n東向島一郵便局\n（株）三菱東京ＵＦＪ銀行 向島支店\n向島郵便...</td>
      <td>\n                    二人入居可\n                 ...</td>
      <td>2年間</td>
      <td>北西</td>
      <td>- / -</td>
      <td>\n                                            ...</td>
      <td>\nサンティーニ\n押上せんべい本舗東向島店\nやきとり大吉東向島店\n栄堂\nとんかつひろ\n</td>
      <td>\n墨田幼稚園\n第一寺島小学校\n寺島中学校\n東京都立 墨田川高等学校\n千葉工業大学東...</td>
      <td>\n                    オートロック\n                ...</td>
      <td>\n                    クローゼット\n                ...</td>
      <td>2階/-</td>
      <td>2階 / 12階建\n</td>
      <td>\n    1K\n     ( キッチン 2帖(2階)\n洋室 8.5帖(2階) )</td>
      <td>26.56m²</td>
      <td>1</td>
      <td>要</td>
      <td>\n済生会向島病院\n中林病院\n健生堂病院\n向島医院\n台東区立台東病院\n</td>
      <td>2019年11月 ( 新築 )</td>
      <td>\n                    コンロ二口\n                 ...</td>
      <td>\n                    二人入居可\n                 ...</td>
      <td>1</td>
      <td>RELUXIA東向島</td>
      <td>1</td>
      <td>徒歩10分以内（2駅）</td>
      <td>すべての駅（3駅）</td>
      <td>\n                                    　新築・インター...</td>
      <td>1</td>
      <td>1</td>
      <td>\n東京都 建設局 向島百花園サービスセンター\n台東区役所 台東区立図書館 石浜\n浅草２...</td>
      <td>仲介手数料半額</td>
      <td>0</td>
      <td>新賃料の1ヶ月分</td>
      <td>8.5万円</td>
      <td>8.5万円 ( 10,000円 )</td>
      <td>無 / 1ヶ月</td>
      <td>1</td>
      <td>\nセブン－イレブン 墨田東向島４丁目店\nグルメシティー東向島駅前店\nあおぞら薬局\n柳...</td>
      <td>0</td>
      <td>東武伊勢崎線 曳舟駅 徒歩6分普通区間準急準急区間急行急行京成押上線 京成曳舟駅 徒歩7分普...</td>
      <td>未完成</td>
      <td>\n                                            ...</td>
      <td>\n東武伊勢崎線 曳舟駅 徒歩6分\n京成押上線 京成曳舟駅 徒歩7分\n東武伊勢崎線 東向...</td>
      <td>https://www.homes.co.jp/chintai/b-1313570020198/</td>
    </tr>
    <tr>
      <th>2</th>
      <td>東京都墨田区東向島3丁目</td>
      <td>1</td>
      <td>1</td>
      <td>-</td>
      <td>\n東京東信用金庫 本店\n向島郵便局\n（株）三菱東京ＵＦＪ銀行 向島支店\n東向島一郵便...</td>
      <td>\n                    ペット相談\n                 ...</td>
      <td>2年間</td>
      <td>北西</td>
      <td>- / -</td>
      <td>\n                                            ...</td>
      <td>\n鳩家\nスイートサンクチュアリーイソ\nそれいゆさんさん\n押上せんべい本舗東向島店\n...</td>
      <td>\n第三寺島幼稚園\n第一寺島小学校\n寺島中学校\n東京都立 墨田川高等学校\n千葉工業大...</td>
      <td>\n                    オートロック\n                ...</td>
      <td>\n                    クローゼット\n                ...</td>
      <td>3階/-</td>
      <td>3階 / 12階建\n</td>
      <td>\n    1K\n     ( 洋室 8.5帖\nキッチン 2帖 )</td>
      <td>26.56m²</td>
      <td>1</td>
      <td>要</td>
      <td>\n済生会向島病院\n中西整形外科内科\n健生堂病院\nアップル歯科\n賛育会病院健康管理ク...</td>
      <td>2019年11月 ( 新築 )</td>
      <td>\n                    コンロ二口\n                 ...</td>
      <td>\n                    ペット相談\n                 ...</td>
      <td>1</td>
      <td>RELUXIA東向島</td>
      <td>1</td>
      <td>徒歩10分以内（2駅）</td>
      <td>すべての駅（2駅）</td>
      <td>\n                                    家賃・仲介手数料...</td>
      <td>1</td>
      <td>1</td>
      <td>\n東京都 建設局 向島百花園サービスセンター\n墨田区立 ひきふね図書館\n浅草２丁目町会...</td>
      <td>新築分譲ペット相談可インターネット無料</td>
      <td>0</td>
      <td>新賃料の1ヶ月分</td>
      <td>8.55万円</td>
      <td>8.55万円 ( 10,000円 )</td>
      <td>無 / 1ヶ月</td>
      <td>1</td>
      <td>\nローソン 墨田東向島二丁目店\nまいばすけっと 東武曳舟駅西店\nさつき薬局\n菊屋本店...</td>
      <td>0</td>
      <td>東武伊勢崎線 曳舟駅 徒歩6分普通区間準急準急区間急行急行京成押上線 京成曳舟駅 徒歩7分普...</td>
      <td>未完成</td>
      <td>\n                                            ...</td>
      <td>\n東武伊勢崎線 曳舟駅 徒歩6分\n京成押上線 京成曳舟駅 徒歩7分\n\n通勤・通学駅ま...</td>
      <td>https://www.homes.co.jp/chintai/b-1171750025121/</td>
    </tr>
    <tr>
      <th>3</th>
      <td>東京都杉並区上荻2丁目</td>
      <td>1</td>
      <td>1</td>
      <td>-</td>
      <td>\n三井住友信託銀行（株） 荻窪支店\n杉並四面道郵便局\n（株）三井住友銀行 荻窪支店\n...</td>
      <td>\n                    コンロ二口\n                 ...</td>
      <td>2年間</td>
      <td>東</td>
      <td>- / -</td>
      <td>\n                                            ...</td>
      <td>\n本むら庵・手打ちそば・荻窪本店\n亀屋菓子（株）北口本店\n魚くま\n田村米菓総本舗\n...</td>
      <td>\nピヨピヨおうちえん\n桃井第二小学校\n神明中学校\n東京都立 荻窪高等学校\n東京女子...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2階/205</td>
      <td>2階 / 3階建\n</td>
      <td>\n    1K\n     ( キッチン 2.7帖\n洋室 7.1帖 )</td>
      <td>26.16m²</td>
      <td>1</td>
      <td>要</td>
      <td>\n東京衛生病院 附属健診センター\nカワイ歯科クリニック\n東京衛生病院 予約センター\n...</td>
      <td>2020年1月 ( 新築 )</td>
      <td>\n                    オートロック\n                ...</td>
      <td>\n                    コンロ二口\n                 ...</td>
      <td>1</td>
      <td>Ｐｒｅｍｉａｓ荻窪</td>
      <td>1</td>
      <td>徒歩10分以内（1駅）</td>
      <td>すべての駅（1駅）</td>
      <td>\n                                    施工会社：積水ハ...</td>
      <td>0</td>
      <td>0</td>
      <td>\n公園管理事務所 大田黒公園管理事務所\n杉並区立 西荻図書館\n杉並区役所 西荻南区民集...</td>
      <td>ＪＲ中央線荻窪駅より徒歩8分（640ｍ）の好立地4線利用可能で都心部へ快適アクセスエントラン</td>
      <td>1</td>
      <td>新賃料の1ヶ月分</td>
      <td>9.9万円</td>
      <td>9.9万円 ( 7,000円 )</td>
      <td>1ヶ月 / 1ヶ月</td>
      <td>1</td>
      <td>\nファミリーマート 杉並上荻二丁目店\nウッディーワン\n（有）ステラ薬局\n井上茶舗\n...</td>
      <td>0</td>
      <td>ＪＲ中央線 荻窪駅 徒歩8分普通快速通勤快速通勤特別快速中央特快青梅特快</td>
      <td>未完成</td>
      <td>\n                                            ...</td>
      <td>\nＪＲ中央線 荻窪駅 徒歩8分\n\n通勤・通学駅までの経路・所要時間を調べる\n\n</td>
      <td>https://www.homes.co.jp/chintai/b-37031260003000/</td>
    </tr>
    <tr>
      <th>4</th>
      <td>東京都杉並区上荻2丁目</td>
      <td>1</td>
      <td>1</td>
      <td>-</td>
      <td>\n三井住友信託銀行（株） 荻窪支店\n杉並四面道郵便局\n（株）三井住友銀行 荻窪支店\n...</td>
      <td>\n                                            ...</td>
      <td>2年間</td>
      <td>西</td>
      <td>- / -</td>
      <td>\n                                            ...</td>
      <td>\n本むら庵・手打ちそば・荻窪本店\n亀屋菓子（株）北口本店\n魚くま\n田村米菓総本舗\n...</td>
      <td>\nピヨピヨおうちえん\n桃井第二小学校\n神明中学校\n東京都立 荻窪高等学校\n東京女子...</td>
      <td>\n                    オートロック\n                ...</td>
      <td>NaN</td>
      <td>2階/201</td>
      <td>2階 / 3階建\n</td>
      <td>\n    1K\n     ( キッチン 1.8帖\n洋室 7.5帖 )</td>
      <td>26.48m²</td>
      <td>1</td>
      <td>要</td>
      <td>\n東京衛生病院 附属健診センター\nカワイ歯科クリニック\n東京衛生病院 予約センター\n...</td>
      <td>2020年1月 ( 新築 )</td>
      <td>\n                    コンロ二口\n                 ...</td>
      <td>\n                                            ...</td>
      <td>1</td>
      <td>Ｐｒｅｍｉａｓ荻窪</td>
      <td>1</td>
      <td>徒歩10分以内（1駅）</td>
      <td>すべての駅（1駅）</td>
      <td>\n                                    施工会社：積水ハ...</td>
      <td>0</td>
      <td>0</td>
      <td>\n公園管理事務所 大田黒公園管理事務所\n杉並区立 西荻図書館\n杉並区役所 西荻南区民集...</td>
      <td>ＪＲ中央線荻窪駅より徒歩8分（640ｍ）の好立地4線利用可能で都心部へ快適アクセスエントラン</td>
      <td>1</td>
      <td>新賃料の1ヶ月分</td>
      <td>9.9万円</td>
      <td>9.9万円 ( 7,000円 )</td>
      <td>1ヶ月 / 1ヶ月</td>
      <td>1</td>
      <td>\nファミリーマート 杉並上荻二丁目店\nウッディーワン\n（有）ステラ薬局\n井上茶舗\n...</td>
      <td>0</td>
      <td>ＪＲ中央線 荻窪駅 徒歩8分普通快速通勤快速通勤特別快速中央特快青梅特快</td>
      <td>未完成</td>
      <td>\n                                            ...</td>
      <td>\nＪＲ中央線 荻窪駅 徒歩8分\n\n通勤・通学駅までの経路・所要時間を調べる\n\n</td>
      <td>https://www.homes.co.jp/chintai/b-37031260003010/</td>
    </tr>
  </tbody>
</table>
</div>



## Data description
- `name` - 物件名
- `floor` - 階数
- `rent` - 家賃
- `rent_all` - 家賃（共益費）
- `security_deposit ` - 敷金・礼金
- `deposit` - 保証金
- `traffic` - 最寄り駅などの情報
- `address` - 住所
- `number_of_stations_10_min` - 徒歩10分以内の駅数
- `number_of_stations_all` - 近くにある全ての駅の数
- `station_express_info` - 近くの駅の停車する電車の情報（急行・快速など）
- `shopping` - 近くにあるショッピング施設
- `eatting`- 近くにある飲食店
- `education` - 近くにある教育施設
- `hospital`- 近くにある病院
- `bank` - 近くにある銀行
- `public_facility` - 近くにある公共施設
- `how_old` - 築年数
- `daylight_direction` - 採光面
- `floor_space` - 床面積
- `balcony_space` - バルコニーの面積
- `floor_plan` - 間取り
- `recomend_point` - おすすめポイント
- `contract_period` - 契約期間
- `renewal_fee` - 更新料
- `deposit2` - 保証金2
- `home_insurance` - 住宅保険
- `status` - 状態（空き家など）
- `conditions` - 条件
- `kichen` - キッチン等の情報
- `equipments` - 設備の情報
- `other` - その他の情報
- `url` - 物件のURL

## Data cleaning
EDAやモデルの学習をおこなうために、データクリーニングをします。  
具体的には、以下の処理をしました。
- target変数である`rent`が欠損している行は削除
- すべてobject型になっているので、家賃など数値として扱うものは整数型に変更


```python
df_all.dropna(subset=['rent'], inplace=True)
```


```python
# 共益費
df_all['service_fee'] = \
    df_all.apply(lambda row: row['rent_all'].lstrip(row['rent']).strip(' ()円').replace(',','').replace('-', '0'), axis=1).astype('int')
df_all.drop('rent_all', axis=1, inplace=True)
```


```python
df_all['rent'] = df_all['rent'].apply(lambda x: float(x.rstrip('万円'))*10000).astype('int')
```

## EDA and sample statistics
データの特性を知り、特徴量エンジニアリングやモデルの選定の際に活かします。

### `rent`（家賃）の分布と統計量


```python
plt.figure(figsize=(12, 8))
plt.hist(df_all['rent'], bins=400)
plt.xlim(0, 500000)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_15_0.png)



```python
df_all['rent'].describe()
```




    count    1.646850e+05
    mean     8.504018e+04
    std      4.269391e+04
    min      1.050000e+04
    25%      6.000000e+04
    50%      7.500000e+04
    75%      9.900000e+04
    max      1.550000e+06
    Name: rent, dtype: float64



平均が91652円で、中央値が85000円。さすが東京、高いですね。。

### `service_fee`（共益費）の分布


```python
plt.figure(figsize=(12, 8))
plt.hist(df_all['service_fee'], bins=100)
plt.xlim(0, 50000)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_19_0.png)



```python
df_all['service_fee'].describe()
```




    count    164685.000000
    mean       4407.091289
    std        3983.448083
    min           0.000000
    25%        2000.000000
    50%        3000.000000
    75%        6000.000000
    max      150000.000000
    Name: service_fee, dtype: float64



共益費はあんまり綺麗な分布ではないですね。

### 目的変数の設定
このデータ分析では`rent`（家賃）と`service_fee`（共益費）を合計した金額を目的変数とします。  


```python
# 目的変数
df_all['target'] = df_all['rent'] + df_all['service_fee']
```


```python
plt.figure(figsize=(12, 8))
plt.hist(df_all['target'], bins=400)
plt.xlim(0, 500000)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_24_0.png)


右裾が長い分布になっています。  
どれくらい正規分布からズレているか定量的に測ってみましょう。正規分布からのズレ具合を表す統計量として**歪度**があります。  
歪度は`Scipy`を使うことで簡単に計算できます。


```python
import scipy

scipy.stats.skew(df_all['target'])
```




    3.0333404245855253



歪度は1以上だとひどく歪んだ分布なので、東京都の家賃分布はかなり裾が長い分布と言えます。  
田舎だと家賃の幅が狭くて、歪度はもう少し低い値になりそうですね。  

**損失関数によっては、このような正規分布から大きく外れた分布は不都合が生じることがあるため、分布を正規分布に近い形に変換する必要があります。**

### 目的変数の対数変換
目的変数を対数変換することで、分布を正規分布に近づけます。


```python
import numpy as np

plt.figure(figsize=(12, 8))
plt.hist(np.log1p(df_all['target']), bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_29_0.png)


見た目上、正規分布に近づいていることがわかります。定量的に確認するために、歪度をもう一度計算します。


```python
scipy.stats.skew(np.log1p(df_all['target']))
```




    0.3252554367056249



対数変換によって歪度が大きく減っている（正規分布に近づいていること）が定量的に言えます。（参考：正規分布だと歪度は0になります）

### 徒歩10分圏内の駅の数


```python
df_all['number_of_stations_10_min'].unique()
```




    array(['徒歩10分以内（1駅）', '徒歩10分以内（2駅）', '徒歩10分以内（0駅）', nan, '徒歩10分以内（4駅）',
           '徒歩10分以内（6駅）', '徒歩10分以内（3駅）', '徒歩10分以内（5駅）'], dtype=object)



駅数が欠損しているデータがあります。欠損している行を確認してみます。


```python
df_all[df_all['number_of_stations_10_min'].isnull()].head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
      <th>air_conditioner</th>
      <th>auto_lock</th>
      <th>balcony_space</th>
      <th>bank</th>
      <th>conditions</th>
      <th>contract_period</th>
      <th>daylight_direction</th>
      <th>deposit</th>
      <th>deposit2</th>
      <th>eatting</th>
      <th>education</th>
      <th>equipments</th>
      <th>equipments2</th>
      <th>floor</th>
      <th>floor2</th>
      <th>floor_plan</th>
      <th>floor_space</th>
      <th>flooring</th>
      <th>home_insurance</th>
      <th>hospital</th>
      <th>how_old</th>
      <th>kichen</th>
      <th>location</th>
      <th>more_than_2</th>
      <th>name</th>
      <th>new_house</th>
      <th>number_of_stations_10_min</th>
      <th>number_of_stations_all</th>
      <th>other</th>
      <th>parking</th>
      <th>pets</th>
      <th>public_facility</th>
      <th>recomend_point</th>
      <th>reheating</th>
      <th>renewal_fee</th>
      <th>rent</th>
      <th>security_deposit</th>
      <th>separate</th>
      <th>shopping</th>
      <th>south</th>
      <th>station_express_info</th>
      <th>status</th>
      <th>structure</th>
      <th>traffic</th>
      <th>url</th>
      <th>service_fee</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>東京都府中市是政3丁目64-9</td>
      <td>1</td>
      <td>0</td>
      <td>-</td>
      <td>NaN</td>
      <td>\n                    二人入居可\n                 ...</td>
      <td>2年間</td>
      <td>南東</td>
      <td>- / -</td>
      <td>\n                                            ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>\n                    TVモニタ付インターホン\n          ...</td>
      <td>\n                                            ...</td>
      <td>1階/102</td>
      <td>1階 / 2階建\n</td>
      <td>\n    1LDK\n     ( ダイニングキッチン 6.2帖\n洋室 5.1帖 )</td>
      <td>29.75m²</td>
      <td>1</td>
      <td>要</td>
      <td>NaN</td>
      <td>2019年9月 ( 新築 )</td>
      <td>\n                    システムキッチン\n              ...</td>
      <td>\n                    二人入居可\n                 ...</td>
      <td>0</td>
      <td>ファインコーラル</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>\n                                    保証会社必須：初...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>モニター付きインターフォン・浴室乾燥機・追い焚き・温水洗浄便座・独立洗面台</td>
      <td>1</td>
      <td>新賃料の1ヶ月分</td>
      <td>74000</td>
      <td>1ヶ月 / 無</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>ＪＲ南武線 府中本町駅 徒歩11分普通快速西武多摩川線 是政駅 徒歩12分普通</td>
      <td>空家</td>
      <td>\n                                            ...</td>
      <td>\nＪＲ南武線 府中本町駅 徒歩11分\n西武多摩川線 是政駅 徒歩12分\n\n通勤・通学...</td>
      <td>https://www.homes.co.jp/chintai/b-1142840600220/</td>
      <td>3000</td>
      <td>77000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>東京都文京区本駒込2丁目11</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>NaN</td>
      <td>\n                    二人入居可\n                 ...</td>
      <td>2年間</td>
      <td>南東</td>
      <td>- / -</td>
      <td>\n                                            ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>\n                    オートロック\n                ...</td>
      <td>\n                    全居室収納\n                 ...</td>
      <td>2階/206</td>
      <td>2階 / 3階建\n</td>
      <td>\n    2LDK\n     ( 洋室 4.5帖\n洋室 6.8帖\nリビングダイニング...</td>
      <td>56.3m²</td>
      <td>1</td>
      <td>要</td>
      <td>NaN</td>
      <td>2019年10月 ( 築1年 )</td>
      <td>\n                    コンロ三口\n                 ...</td>
      <td>\n                    二人入居可\n                 ...</td>
      <td>1</td>
      <td>レイ　レ　ラヴィ</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>\n                                    お電話でのお問い...</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>新築物件 即日案内可能 角部屋２面採光 ネット無料 ペット飼育可能</td>
      <td>1</td>
      <td>新賃料の1ヶ月分</td>
      <td>158000</td>
      <td>1ヶ月 / 1ヶ月</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>都営三田線 千石駅 徒歩3分普通急行東京メトロ南北線 本駒込駅 徒歩9分普通</td>
      <td>空家</td>
      <td>\n                                            ...</td>
      <td>\n都営三田線 千石駅 徒歩3分\n東京メトロ南北線 本駒込駅 徒歩9分\n都営三田線白山駅...</td>
      <td>https://www.homes.co.jp/chintai/b-1408470000244/</td>
      <td>4000</td>
      <td>162000</td>
    </tr>
  </tbody>
</table>
</div>



駅数が欠損していても、`traffic`から徒歩10分圏内の駅数は0じゃないことがわかります。

### `number_of_stations_10_min`と`number_of_stations_all`の欠損値の補完
欠損していない`traffic`から文字を抽出して、欠損している駅数を補完します。


```python
df_all['traffic'].iloc[0]
```




    '\n西武新宿線 新井薬師前駅 徒歩4分\nＪＲ中央線 中野駅 徒歩18分\n\n通勤・通学駅までの経路・所要時間を調べる\n\n'



上の文から10分以内の駅数、家の近くのすべての駅数を抽出できそうです。


```python
import re

def count_10min_stations(col):
    stations = 0
    for i in col.splitlines():
        if 'バス' not in i:
            stations += len(re.findall('徒歩[0-9]分|徒歩10分', i))
    return stations

def count_near_stations(col):
    # ??駅 徒歩xx分という文の数をカウントして、駅数を求める。
    stations = len(re.findall('徒歩\d+分', col))
    try:
        # 他に??駅が利用可能という文から??駅を抽出
        stations += int(re.findall('に\d+駅', col)[0].strip('駅に'))
    except:
        # 他に??駅が利用可能という文が存在しない場合、配列外参照エラーがでるため例外処理しています。
        pass
    return stations
```


```python
# 徒歩10分以内の駅数
# df_all['number_of_stations_10_min'] = df_all['traffic'].apply(lambda x: len(re.findall('徒歩[0-9]|10分', x))).astype('int')
df_all['number_of_stations_10_min'] = df_all['traffic'].apply(count_10min_stations).astype('int')
# 家の近くのすべての駅数
df_all['number_of_stations_all'] = df_all['traffic'].apply(count_near_stations).astype('int')
```


```python
plt.figure(figsize=(12, 8))
plt.hist(df_all['number_of_stations_10_min'], bins=30)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_43_0.png)



```python
df_all['number_of_stations_10_min'].value_counts()
```




    1    67601
    0    46957
    2    37202
    3    12612
    4      311
    5        2
    Name: number_of_stations_10_min, dtype: int64




```python
plt.figure(figsize=(12, 8))
plt.hist(df_all['number_of_stations_all'], bins=30)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_45_0.png)



```python
df_all['number_of_stations_all'].value_counts()
```




    3     58456
    2     44683
    6     18414
    4     16810
    5     13067
    1     11577
    7      1627
    0        44
    8         4
    10        2
    11        1
    Name: number_of_stations_all, dtype: int64



### 1,2,3番目に近い駅を抽出


```python
df_all.reset_index(drop=True, inplace=True)
```


```python
first_near_station = [None for s in range(len(df_all))]
second_near_station = [None for s in range(len(df_all))]
third_near_station = [None for s in range(len(df_all))]

for index in df_all.index:
    stations_list = [s for s in df_all.loc[index, 'traffic'].splitlines() if ('通勤・通学駅' not in s) and ('他に' not in s) and s]
    for i, station in enumerate(stations_list):
        if i==0:
            # index=8の国立のように国立駅の駅が抜けている場合があるため例外処理
            try:
                first_near_station[index] = re.findall('.*駅', station)[0]
            except:
                first_near_station[index] = None
        elif i==1:
            try:
                second_near_station[index] = re.findall('.*駅', station)[0]
            except:
                second_near_station[index] = None
        elif i==2:
            try:
                third_near_station[index] = re.findall('.*駅', station)[0]
            except:
                third_near_station[index] = None

df_all['first_near_station'] = first_near_station
df_all['second_near_station'] = second_near_station
df_all['third_near_station'] = third_near_station
```

### 床面積


```python
df_all['floor_space'] = df_all['floor_space'].apply(lambda x: x.rstrip('m²').replace(',', '')).astype('float')
```

### 築年数


```python
# 築0年のときは新築と記載されているので、築0年に置き換える。
df_all['how_old'] = df_all['how_old'].apply(lambda x: re.findall(' 築(\d+)年 ', x.replace('新築', '築0年'))[0]).astype('int')
```


```python
plt.figure(figsize=(14,8))
plt.hist(df_all['how_old'], bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_54_0.png)


### 敷金・礼金
敷金・礼金をそのまま使うとデータリークするため、金額が記載されている場合は、何か月分の家賃かに置換する。


```python
df_all['key_money'] = df_all['security_deposit'].apply(lambda x: x.split(' / ')[1])
df_all['security_deposit'] = df_all['security_deposit'].apply(lambda x: x.split(' / ')[0])
```


```python
def preprocess_security_deposit(row):
    if 'ヶ月' in row['security_deposit']:
        # 正規表現で小数点を含む数字を取得
        how_months = float(re.findall('(\d*[.,]?\d*)ヶ月', row['security_deposit'])[0])
    elif '万円' in row['security_deposit']:
        money = float(re.findall('(\d*[.,]?\d*)万円', row['security_deposit'])[0])*10000
        how_months = money/row['rent']
    elif '無'== row['security_deposit']:
        how_months = 0
    else:
        raise Exception
    return how_months

def preprocess_key_money(row):
    if 'ヶ月' in row['key_money']:
        how_months = float(re.findall('(\d*[.,]?\d*)ヶ月', row['key_money'])[0])
    elif '万円' in row['key_money']:
        money = float(re.findall('(\d*[.,]?\d*)万円', row['key_money'])[0])*10000
        how_months = money/row['rent']
    elif '無'== row['key_money']:
        how_months = 0
    else:
        raise Exception
    return how_months

df_all['security_deposit'] = df_all.apply(preprocess_security_deposit, axis=1)
df_all['key_money'] = df_all.apply(preprocess_key_money, axis=1)
```

### 更新料


```python
def preprocess_renewal_fee(row):
    # nanを含んでいる場合、文字型でないとエラーがでる。
    if 'ヶ月' in str(row['renewal_fee']):
        how_months = float(re.findall('新賃料の(\d*[.,]?\d*)ヶ月分', row['renewal_fee'])[0])
    elif '円' in str(row['renewal_fee']):
        money = float(re.sub(',|円', '', row['renewal_fee']))
        how_months = money/row['rent']
    else:
        how_months = 0
    return how_months

df_all['renewal_fee'] = df_all.apply(preprocess_renewal_fee, axis=1)
```

### 最寄り駅までの時間


```python
df_all['time_to_go_nearest_station'] = \
        df_all['traffic'].apply(lambda x: re.findall('徒歩(\d+)分', x)[0] if '徒歩' in x else None).astype('float')
```

### 階数

301階なるものが存在しますが、中国で構想段階の300階建てのビル（1228m）に匹敵する高さなので明らかに間違いです笑  
3階になおしておきます。


```python
df_all[df_all['name']=='ロイヤルコートＫａｙＡＢ'].floor
```




    38283    301階/B206
    Name: floor, dtype: object




```python
df_all.loc[df_all[df_all['name']=='ロイヤルコートＫａｙＡＢ'].index, 'floor'] = '3階'
```


```python
df_all[['floor2','floor']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>floor2</th>
      <th>floor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4階 / 4階建\n</td>
      <td>4階/401</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2階 / 12階建\n</td>
      <td>2階/-</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3階 / 12階建\n</td>
      <td>3階/-</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2階 / 3階建\n</td>
      <td>2階/205</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2階 / 3階建\n</td>
      <td>2階/201</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>164680</th>
      <td>4階 / 9階建\n</td>
      <td>4階/-</td>
    </tr>
    <tr>
      <th>164681</th>
      <td>2階 / 9階建\n</td>
      <td>2階/-</td>
    </tr>
    <tr>
      <th>164682</th>
      <td>1階 / 2階建\n</td>
      <td>1階/-</td>
    </tr>
    <tr>
      <th>164683</th>
      <td>2階 / 2階建\n</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>164684</th>
      <td>2階 / 2階建\n</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>164685 rows × 2 columns</p>
</div>




```python
def extract_max_foor(row):
    try:
        if '階' in str(row):
            max_floor = re.findall('(\d+)階', row)[1]
        else:
            max_floor = None
    except:
        max_floor = None
    return max_floor
```


```python
df_all['max_floor'] = df_all['floor2'].apply(extract_max_foor).astype('float')
```


```python
df_all['floor'] = df_all['floor2'].apply(lambda x: re.findall('(\d+)階', x)[0] if '階' in str(x) else 0).astype('int')
```


```python
df_all.drop(['rent', 'service_fee'], axis=1, inplace=True)
```


```python
mask = np.zeros_like(df_all.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (15,12))
sns.heatmap(df_all.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20)
plt.tight_layout()
plt.ylim(0,len(df_all.corr()))
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_71_0.png)


### ヒートマップからわかること
- `auto_lock`と家賃の相関がかなり大きい。ただし、`auto_lock`は築年数と大きな負の相関があり、これは`auto_lock`が導入され始めたのが比較的新しいためだと考えられる。  
したがって、家賃とオートロックは疑似相関（築年数を介した）の可能性が高い

### 間取り
|アルファベット  |意味  |
|---|---|
|R  |一般的に1部屋の中にキッチンが含まれている間取り  |
|K  |居室とキッチンの間に間仕切りがある間取り  |
|D  |キッチンで食事がとれるようなスペースがある  |
|L  |食事やテレビを見たりできるようなダイニングよりもさらに広いスペースがある  |
|S  |サービスルームの略。採光が不足し居室とは認められないがフリースペースとして使える部屋  |

参考：https://suumo.jp/yougo/m/madori/


```python
df_all['floor_plan'] = df_all['floor_plan'].apply(lambda x: x.split()[0])
```


```python
df_all['floor_plan'].unique()
```




    array(['ワンルーム', '1K', '1LDK', '1DK', '3LDK', '2K', '2LDK', '2DK', '2SLDK',
           '1SK', '1SLDK', '3DK', '1SDK', '1SLK', '3K', '2SDK', '1LK',
           '3SLDK', '4DK', '4LDK', '4SLDK', '3SDK', '10K', '2LK', '5SLDK',
           '5DK', '2R', '4K', '2SLK', '5K', '2SK', '5LDK', '3LK', '6R', '15K',
           '3SK', '5SK', '4SDK', '11K', '8LDK', '6LDK', '7LDK', '5SDK',
           '6SLDK', '3SLK', '6DK', '12DK', '7SLDK', '4SK'], dtype=object)



### 住所


```python
df_all['address_town'] = df_all['address'].apply(lambda x: re.split('\d+', x)[0])
```


```python
numeric_col = [s for s in df_all.columns if df_all[s].dtype != 'object']
object_col = ['floor_plan', 'first_near_station', 'second_near_station', 'third_near_station', 'status', 'address', 'address_town', 'name', 'url', 'structure']
```


```python
df = df_all[numeric_col + object_col]
```


```python
df['target'] = np.log1p(df['target'].copy())
```


```python
plt.figure(figsize=(12, 8))
plt.hist(np.ceil(df_all['target']), bins=20)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_81_0.png)



```python
import geopandas as gpd

# get from https://github.com/dataofjapan/land
df_tokyomap = gpd.read_file('../data/tokyo.geojson')
```


```python
def extract_city_name(row):
    if re.findall('東京都(.+区)', row):
        return re.findall('東京都(.+区)', row)[0]
    elif re.findall('東京都(.+市)', row):
        return re.findall('東京都(.+市)', row)[0]
    elif re.findall('東京都(.+郡)', row):
        return re.findall('東京都(.+郡)', row)[0]
```


```python
df_all['city'] = df_all['address'].apply(extract_city_name)
```


```python
df_toshin = df_tokyomap[df_tokyomap['area_en'] == 'Tokubu'].copy()
```


```python
df_toshin['mean'] = [df_all.groupby('city').mean()[['target']].loc[s].target for s in df_toshin.ward_ja]
```


```python
df_toshin['coords'] = df_toshin.geometry.apply(lambda x: x.representative_point().coords[:][0])
```


```python
df_toshin.plot(column='mean', figsize=(20,12), legend=True, legend_kwds={'label': "Mean rent"}, cmap='viridis')
for idx, row in df_toshin.iterrows():
    plt.annotate(s=row['ward_en'], xy=row['coords'],
                 horizontalalignment='center', color='white')
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_88_0.png)



```python
from sklearn.preprocessing import LabelEncoder

encoder_dict = {}
for col in object_col:
    if col != 'url':
        df[col].fillna('-999', inplace=True)
        encoder_dict[col] = LabelEncoder().fit(df[col])
        df[col] = encoder_dict[col].transform(df[col].copy())
```

# 機械学習モデルの作成
---


```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

X = df.drop('target', axis=1)
y = df['target'].values
for train_idx, test_idx in gss.split(X, y, groups=X['name'].values):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]    
```


```python
# from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
nfold = 3
# folds = KFold(n_splits=nfold, shuffle=True, random_state=42)
folds = GroupKFold(n_splits=nfold)
groups = X_train['name'].values
print('-'*20)
print(str(nfold) + ' Folds training...')
print('-'*20)
```

    --------------------
    3 Folds training...
    --------------------



```python
params_lgb = {
    "boosting": "gbdt",
    "verbosity": -1,
    "num_leaves": 200,
#     "min_data_in_leaf": 10,
    "min_child_weight": 1,
    "max_depth": 8,
    "colsample_bytree": 1.0,
    "subsample": 0.9,
    "gamma": 0,
    "lambda_l2": 1,
    "lambda_l1": 0,
    "learning_rate": 0.1,
    "random_seed": 42,
    "metric": "rmse",
}
```


```python
%%time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

oof = np.zeros(len(X_train))
feature_importance_df = pd.DataFrame()
df_fold_all = pd.DataFrame()
tr_rmse = []
val_rmse = []
models = []

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train, groups)):
    strLog = f"fold {fold_}"
    print(strLog)
    df_fold_id = pd.DataFrame(data={'index':val_idx})
    df_fold_id['folds'] = fold_
    df_fold_all = pd.concat([df_fold_all, df_fold_id])
    
    X_tr, X_val = X_train.drop(['url', 'name'], axis=1).iloc[trn_idx], X_train.drop(['url', 'name'], axis=1).iloc[val_idx]
    y_tr, y_val = y_train[trn_idx], y_train[val_idx]
    
    model = lgb.LGBMRegressor(**params_lgb, n_estimators = 100000, n_jobs = -1)
    model.fit(X_tr, 
              y_tr, 
              eval_set=[(X_tr, y_tr), (X_val, y_val)], 
              eval_metric='rmse',
              verbose=500, 
              early_stopping_rounds=500,
             )

    oof[val_idx] = model.predict(X_val)
    val_score = np.sqrt(mean_squared_error(y_val, oof[val_idx]))
    val_rmse.append(val_score)
    tr_score = np.sqrt(mean_squared_error(y_tr, model.predict(X_tr)))
    tr_rmse.append(tr_score)
    models.append(model)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = X_tr.columns
    fold_importance_df["importance"] = model.booster_.feature_importance(importance_type='gain')[:len(X_tr.columns)]
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

rmse = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(oof)))
df_fold_all.set_index('index', inplace=True)
print(f'RMSE {rmse}円')
```

    fold 0
    Training until validation scores don't improve for 500 rounds
    [500]	training's rmse: 0.0631679	valid_1's rmse: 0.0994069
    [1000]	training's rmse: 0.0493903	valid_1's rmse: 0.0966779
    [1500]	training's rmse: 0.0408242	valid_1's rmse: 0.0956802
    [2000]	training's rmse: 0.0347755	valid_1's rmse: 0.0951268
    [2500]	training's rmse: 0.0299788	valid_1's rmse: 0.0947458
    [3000]	training's rmse: 0.0262723	valid_1's rmse: 0.0945453
    [3500]	training's rmse: 0.0232082	valid_1's rmse: 0.0943761
    [4000]	training's rmse: 0.0208753	valid_1's rmse: 0.094286
    [4500]	training's rmse: 0.0188477	valid_1's rmse: 0.0942086
    [5000]	training's rmse: 0.0172252	valid_1's rmse: 0.0941509
    [5500]	training's rmse: 0.0157543	valid_1's rmse: 0.094122
    [6000]	training's rmse: 0.0145479	valid_1's rmse: 0.0941064
    [6500]	training's rmse: 0.0134946	valid_1's rmse: 0.0940936
    Early stopping, best iteration is:
    [6491]	training's rmse: 0.0135106	valid_1's rmse: 0.0940901
    fold 1
    Training until validation scores don't improve for 500 rounds
    [500]	training's rmse: 0.0629458	valid_1's rmse: 0.0986536
    [1000]	training's rmse: 0.0488432	valid_1's rmse: 0.0956959
    [1500]	training's rmse: 0.0401343	valid_1's rmse: 0.0944739
    [2000]	training's rmse: 0.0341508	valid_1's rmse: 0.0939071
    [2500]	training's rmse: 0.0292898	valid_1's rmse: 0.0936114
    [3000]	training's rmse: 0.0255576	valid_1's rmse: 0.0934163
    [3500]	training's rmse: 0.0226694	valid_1's rmse: 0.0932989
    [4000]	training's rmse: 0.0202343	valid_1's rmse: 0.0932437
    [4500]	training's rmse: 0.0181566	valid_1's rmse: 0.0931954
    [5000]	training's rmse: 0.0164717	valid_1's rmse: 0.0931751
    [5500]	training's rmse: 0.0150671	valid_1's rmse: 0.0931688
    [6000]	training's rmse: 0.0138995	valid_1's rmse: 0.09316
    Early stopping, best iteration is:
    [5692]	training's rmse: 0.01457	valid_1's rmse: 0.0931504
    fold 2
    Training until validation scores don't improve for 500 rounds
    [500]	training's rmse: 0.0630415	valid_1's rmse: 0.100021
    [1000]	training's rmse: 0.0496217	valid_1's rmse: 0.097066
    [1500]	training's rmse: 0.0410849	valid_1's rmse: 0.0957656
    [2000]	training's rmse: 0.0350514	valid_1's rmse: 0.0950447
    [2500]	training's rmse: 0.0302328	valid_1's rmse: 0.0946612
    [3000]	training's rmse: 0.0264548	valid_1's rmse: 0.0943714
    [3500]	training's rmse: 0.0233639	valid_1's rmse: 0.094194
    [4000]	training's rmse: 0.020873	valid_1's rmse: 0.0941034
    [4500]	training's rmse: 0.0187668	valid_1's rmse: 0.09404
    [5000]	training's rmse: 0.0170151	valid_1's rmse: 0.0939978
    [5500]	training's rmse: 0.0156128	valid_1's rmse: 0.0939582
    [6000]	training's rmse: 0.0144372	valid_1's rmse: 0.0939314
    [6500]	training's rmse: 0.0134013	valid_1's rmse: 0.0938994
    [7000]	training's rmse: 0.0124981	valid_1's rmse: 0.0938864
    [7500]	training's rmse: 0.0117596	valid_1's rmse: 0.0938836
    [8000]	training's rmse: 0.011113	valid_1's rmse: 0.0938852
    Early stopping, best iteration is:
    [7508]	training's rmse: 0.011743	valid_1's rmse: 0.0938825



```python
cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:50].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
```


```python
plt.figure(figsize=(6,6))
plt.scatter(np.expm1(oof), np.expm1(y_train), alpha=0.3)
plt.tight_layout()
plt.show()
```


```python
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

## 相対誤差


```python
mean_absolute_percentage_error(np.expm1(y_train), np.expm1(oof))
```


```python
def calc_sc(x):
    return np.round_(50+10*(x-np.average(x))/np.std(x))
```


```python
if model_save:
    for idx, model in enumerate(models):
        model.booster_.save_model(f'../data/lgb_regressor_{idx}.txt')
    X_train = X_train.reset_index(drop=True).join(df_fold_all)
    X_train['target'] = np.expm1(y_train)
    X_train['predict'] = np.expm1(oof)
    X_train['mape'] = X_train.apply(lambda x: (x['predict']-x['target'])/x['target'], axis=1)
    X_train['sc'] = calc_sc(X_train['mape'])
    X_train.to_csv('../data/x_train.csv', index=False)
```


```python
!jupyter nbconvert --to markdown 不動産価格予測.ipynb
!mv 不動産価格予測.md README.md
```


```python
!mv README.md ../
```


```python

```
