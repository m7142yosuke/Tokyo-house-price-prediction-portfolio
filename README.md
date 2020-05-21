# Tokyo house price prediction

最終更新日：2020/05/21  
Yosuke Kobayashi  
m7142yosuke@gmail.com

---
## このnotebookについて
データ分析を目的として保存されていないデータは非常に汚く、予測や可視化をおこなうには前処理が必要です。  
このnotebookでは東京の住宅価格予測を例にスクレイピングで取得した汚い生のデータを加工し、可視化・予測・精度評価するまでの手順の一例を示します。  

## データから分かった意外なこと
* 最寄り駅まで徒歩10分の物件数は徒歩11分の物件数に比べて不自然に多く、できるだけ10分以内に収まるように情報が操作されている可能性がある。幾何学的に考えれば、駅からの距離に比例して駅から同じ時間で行ける物件数は多くなる（物件数が全て同じ密度と仮定）

## 工夫したこと
* 同じ物件情報を使うことによるデータリークを防いだ。例えば、メゾンという建物で1階と5階で物件が掲載されていたとする。この場合、1階を訓練データ、5階をテストデータとすると、お互いに似たや賃貸であるためテストの精度は過剰に楽観的になってしまう。そこで物件名でグルーピングして同じ物件名の情報が訓練データとテストデータに別れないようにした。
* 欠損データの補完。例えば、「徒歩十分以内の駅数」は欠損していても「交通情報」は欠損していない場合が多い。そこで交通情報から正規表現を使って徒歩十分圏内の駅数をカウントして補完した。
* 敷金・礼金などの情報は家賃1ヶ月分の金額であることが多いため、そのまま説明変数として使用するとデータリークにつながる。そこで金額という情報を家賃何ヶ月分かという情報に変換して使用した。
* 目的変数の対数変換。目的変数のヒストグラムは右袖が長い分布である。損失関数によっては正規分布から大きく外れた分布だと不都合が生じるため、対数変換により正規分布に近い分布に変換した。また正規分布に近い分布に変換されたことを歪度を使って定量的に評価した。

## 予測精度について
テストデータに対して相対誤差が約5.9％となり、それなりに良い精度となった。

## データの取得方法
2019年末にスクレイピングによりHOME'Sから取得しました。  
約16万5千軒の情報があります。  
※僕がスクレイピングした時点では利用事項にクローリング等を禁止するような文言は見当たりませんでした。

# データの前処理とEDA
---

## Data description
変数名とその意味
- `name` - 物件名
- `floor` - 階数
- `rent` - 家賃
- `rent_all` - 家賃（共益費）
- `security_deposit ` - 敷金
- `key_money` - 礼金
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
- `new_house` - 新築か否か
- `daylight_direction` - 採光面
- `floor_space` - 床面積
- `balcony_space` - バルコニーの面積
- `floor_plan` - 間取り
- `recomend_point` - おすすめポイント
- `contract_period` - 契約期間
- `renewal_fee` - 更新料
- `deposit2` - 保証金2
- `parking` - 駐車場の有無
- `home_insurance` - 住宅保険
- `status` - 状態（空き家など）
- `pets` - ペットがOKか否か
- `conditions` - 条件
- `kichen` - キッチン等の情報
- `equipments` - 設備の情報
- `structure` - RCなどの建物構造
- `other` - その他の情報
- `separate` - セパレートかどうか
- `url` - 物件のURL

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
import japanize_matplotlib
import pickle
import glob
import warnings
warnings.simplefilter('ignore')
sns.set()

p = Path('./data')
df_all = pd.concat([pd.read_csv(f, index_col=0) for f in p.glob('*.csv')], sort=True)

# 重複した物件を削除
df_all.drop_duplicates(subset='url', keep='first', inplace=True)

pd.set_option('display.max_columns', 100)
```


```python
df_all.head(2)
```




<div>
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
      <td>1.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>要</td>
      <td>\nクリニックヨコヤマ\n大竹歯科医院\n新渡戸記念中野総合病院\n寺内医院\n総合東京病院\n</td>
      <td>2019年7月 ( 新築 )</td>
      <td>\n                    IHコンロ\n                 ...</td>
      <td>\n                                            ...</td>
      <td>1.0</td>
      <td>FARE中野7</td>
      <td>1.0</td>
      <td>徒歩10分以内（1駅）</td>
      <td>すべての駅（2駅）</td>
      <td>\n                                    保証会社要加入保...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>\n中野区役所 公園事務所哲学堂公園事務所\n中野区立 上高田図書館\n豊島区役所 区民ひろ...</td>
      <td>RC造新築デザイナーズ1Rマンション</td>
      <td>1.0</td>
      <td>87,000円</td>
      <td>8.7万円</td>
      <td>8.7万円 ( 5,000円 )</td>
      <td>8.7万円 / 無</td>
      <td>1.0</td>
      <td>\nローソンストア１００中野新井四丁目店\nｍｉｎｉピアゴ新井５丁目店\nあらいやくし薬局\...</td>
      <td>0.0</td>
      <td>西武新宿線 新井薬師前駅 徒歩4分普通準急急行通勤急行ＪＲ中央線 中野駅 徒歩18分普通快速...</td>
      <td>空家</td>
      <td>\n                                            ...</td>
      <td>\n西武新宿線 新井薬師前駅 徒歩4分\nＪＲ中央線 中野駅 徒歩18分\n\n通勤・通学駅...</td>
      <td>https://www.homes.co.jp/chintai/b-1064980018577/</td>
    </tr>
    <tr>
      <th>1</th>
      <td>東京都墨田区東向島3丁目</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>要</td>
      <td>\n済生会向島病院\n中林病院\n健生堂病院\n向島医院\n台東区立台東病院\n</td>
      <td>2019年11月 ( 新築 )</td>
      <td>\n                    コンロ二口\n                 ...</td>
      <td>\n                    二人入居可\n                 ...</td>
      <td>1.0</td>
      <td>RELUXIA東向島</td>
      <td>1.0</td>
      <td>徒歩10分以内（2駅）</td>
      <td>すべての駅（3駅）</td>
      <td>\n                                    　新築・インター...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>\n東京都 建設局 向島百花園サービスセンター\n台東区役所 台東区立図書館 石浜\n浅草２...</td>
      <td>仲介手数料半額</td>
      <td>0.0</td>
      <td>新賃料の1ヶ月分</td>
      <td>8.5万円</td>
      <td>8.5万円 ( 10,000円 )</td>
      <td>無 / 1ヶ月</td>
      <td>1.0</td>
      <td>\nセブン－イレブン 墨田東向島４丁目店\nグルメシティー東向島駅前店\nあおぞら薬局\n柳...</td>
      <td>0.0</td>
      <td>東武伊勢崎線 曳舟駅 徒歩6分普通区間準急準急区間急行急行京成押上線 京成曳舟駅 徒歩7分普...</td>
      <td>未完成</td>
      <td>\n                                            ...</td>
      <td>\n東武伊勢崎線 曳舟駅 徒歩6分\n京成押上線 京成曳舟駅 徒歩7分\n東武伊勢崎線 東向...</td>
      <td>https://www.homes.co.jp/chintai/b-1313570020198/</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_all.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 164717 entries, 0 to 東京都三鷹市上連雀
    Data columns (total 47 columns):
     #   Column                     Non-Null Count   Dtype  
    ---  ------                     --------------   -----  
     0   address                    164685 non-null  object 
     1   air_conditioner            164716 non-null  float64
     2   auto_lock                  164716 non-null  float64
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
     18  flooring                   164716 non-null  float64
     19  home_insurance             164685 non-null  object 
     20  hospital                   159775 non-null  object 
     21  how_old                    164685 non-null  object 
     22  kichen                     164475 non-null  object 
     23  location                   164685 non-null  object 
     24  more_than_2                164716 non-null  float64
     25  name                       164685 non-null  object 
     26  new_house                  164716 non-null  float64
     27  number_of_stations_10_min  160115 non-null  object 
     28  number_of_stations_all     160115 non-null  object 
     29  other                      148259 non-null  object 
     30  parking                    164716 non-null  float64
     31  pets                       164716 non-null  float64
     32  public_facility            159760 non-null  object 
     33  recomend_point             155964 non-null  object 
     34  reheating                  164716 non-null  float64
     35  renewal_fee                139187 non-null  object 
     36  rent                       164685 non-null  object 
     37  rent_all                   164685 non-null  object 
     38  security_deposit           164685 non-null  object 
     39  separate                   164716 non-null  float64
     40  shopping                   159775 non-null  object 
     41  south                      164716 non-null  float64
     42  station_express_info       164676 non-null  object 
     43  status                     164685 non-null  object 
     44  structure                  164684 non-null  object 
     45  traffic                    164685 non-null  object 
     46  url                        164716 non-null  object 
    dtypes: float64(10), object(37)
    memory usage: 60.3+ MB


---
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
sns.distplot(df_all['rent'], bins=400)
plt.xlim(0, 500000)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_14_0.png)



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



平均が91652円で、中央値が85000円。やはり東京の家賃は高いですね。

### `service_fee`（共益費）の分布


```python
plt.figure(figsize=(12, 8))
plt.hist(df_all['service_fee'], bins=100)
plt.xlim(0, 50000)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_18_0.png)



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
このデータ分析では`rent`（家賃）に`service_fee`（共益費）を足した金額を目的変数とします。  


```python
# 目的変数
df_all['target'] = df_all['rent'] + df_all['service_fee']
```


```python
plt.figure(figsize=(12, 8))
sns.distplot(df_all['target'], bins=400)
plt.xlim(0, 500000)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_23_0.png)


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
sns.distplot(np.log1p(df_all['target']), bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_28_0.png)


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
df_all[df_all['number_of_stations_10_min'].isnull()][['number_of_stations_10_min', 'traffic']].head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number_of_stations_10_min</th>
      <th>traffic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>NaN</td>
      <td>\nＪＲ南武線 府中本町駅 徒歩11分\n西武多摩川線 是政駅 徒歩12分\n\n通勤・通学...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>\n都営三田線 千石駅 徒歩3分\n東京メトロ南北線 本駒込駅 徒歩9分\n都営三田線白山駅...</td>
    </tr>
  </tbody>
</table>
</div>



駅数が欠損していても、`traffic`から徒歩10分圏内の駅数は0じゃないことがわかります。  
よって、`traffic`から徒歩10分以内の駅を抽出し、`number_of_stations_10_min`を補完することができます。

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

### 徒歩10分以内にある駅数のヒストグラム


```python
plt.figure(figsize=(12, 8))
plt.hist(df_all['number_of_stations_10_min'], bins=30)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_43_0.png)


### 家の近くにある駅数のヒストグラム


```python
plt.figure(figsize=(12, 8))
plt.hist(df_all['number_of_stations_all'], bins=30)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_45_0.png)


なぜか家の近くの駅数が5駅よりも6駅の方が多いですね。。

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

## 掲載物件数の多い最寄り駅TOP10


```python
df_all.groupby('first_near_station').count().sort_values(by='address', ascending=False)[['address']].head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
    </tr>
    <tr>
      <th>first_near_station</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ＪＲ中央線 八王子駅</th>
      <td>2288</td>
    </tr>
    <tr>
      <th>ＪＲ中央線 西八王子駅</th>
      <td>1794</td>
    </tr>
    <tr>
      <th>東京メトロ東西線 葛西駅</th>
      <td>1685</td>
    </tr>
    <tr>
      <th>ＪＲ総武線 小岩駅</th>
      <td>1475</td>
    </tr>
    <tr>
      <th>ＪＲ中央線 三鷹駅</th>
      <td>1307</td>
    </tr>
    <tr>
      <th>ＪＲ総武線 新小岩駅</th>
      <td>1269</td>
    </tr>
    <tr>
      <th>京王相模原線 京王堀之内駅</th>
      <td>1186</td>
    </tr>
    <tr>
      <th>小田急小田原線 町田駅</th>
      <td>1139</td>
    </tr>
    <tr>
      <th>西武池袋線 大泉学園駅</th>
      <td>1018</td>
    </tr>
    <tr>
      <th>ＪＲ中央線 国分寺駅</th>
      <td>1000</td>
    </tr>
  </tbody>
</table>
</div>



23区外が上位を占めていますね。需要と供給の観点から考えて、区外は供給過多のため家賃が安い、区内は供給不足のため家賃が高いと考えられます。

## 最寄り駅までの時間


```python
df_all['time_to_go_nearest_station'] = \
        df_all['traffic'].apply(lambda x: re.findall('徒歩(\d+)分', x)[0] if '徒歩' in x else None).astype('float')
```

### 最寄り駅までの時間のヒストグラム


```python
plt.figure(figsize=(12, 8))
plt.hist(df_all['time_to_go_nearest_station'], bins=30)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_56_0.png)


## 最寄り駅までの時間TOP20


```python
df_all.groupby('time_to_go_nearest_station').count().sort_values(by='address', ascending=False)[['address']].head(20)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
    </tr>
    <tr>
      <th>time_to_go_nearest_station</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5.0</th>
      <td>18464</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>13669</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>13486</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>13397</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>13269</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>12895</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>11976</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>9882</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>9766</td>
    </tr>
    <tr>
      <th>12.0</th>
      <td>6586</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>5753</td>
    </tr>
    <tr>
      <th>13.0</th>
      <td>5637</td>
    </tr>
    <tr>
      <th>15.0</th>
      <td>5449</td>
    </tr>
    <tr>
      <th>11.0</th>
      <td>5023</td>
    </tr>
    <tr>
      <th>14.0</th>
      <td>4244</td>
    </tr>
    <tr>
      <th>17.0</th>
      <td>2487</td>
    </tr>
    <tr>
      <th>18.0</th>
      <td>2355</td>
    </tr>
    <tr>
      <th>16.0</th>
      <td>2061</td>
    </tr>
    <tr>
      <th>20.0</th>
      <td>1872</td>
    </tr>
    <tr>
      <th>19.0</th>
      <td>1410</td>
    </tr>
  </tbody>
</table>
</div>



最寄り駅までの時間が10分と11分で不自然に隔たりがある。このことから最寄り駅までの時間は正確な時間ではなく、できるだけ10分以内に収まるように操作されている可能性がある。

## 間取り
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

### 間取りのヒストグラム


```python
plt.figure(figsize=(14,8))
plt.hist(df_all['floor_plan'], bins=100)
plt.tick_params(axis='x', labelrotation=90)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_63_0.png)


## 床面積


```python
df_all['floor_space'] = df_all['floor_space'].apply(lambda x: x.rstrip('m²').replace(',', '')).astype('float')
```


```python
df_all['floor_space'].describe()
```




    count    164685.000000
    mean         32.126207
    std          16.606834
    min           1.000000
    25%          20.560000
    50%          26.400000
    75%          40.500000
    max        2422.000000
    Name: floor_space, dtype: float64



### 床面積のヒストグラム


```python
plt.figure(figsize=(14,8))
plt.hist(df_all['floor_space'], bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_68_0.png)


最大の床面積を持つ物件を確認してみます。


```python
df_all[df_all.floor_space==df_all.floor_space.max()][['how_old', 'floor_plan', 'floor_space', 'target']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>how_old</th>
      <th>floor_plan</th>
      <th>floor_space</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138517</th>
      <td>1985年10月 ( 築35年 )</td>
      <td>1K</td>
      <td>2422.0</td>
      <td>53000</td>
    </tr>
  </tbody>
</table>
</div>



築35年の1Kの賃貸が最大の床面積とは考えにくく、床面積の誤入力だと推定できます。  
この物件情報は削除します。


```python
idx = df_all[df_all.floor_space==df_all.floor_space.max()][['how_old', 'floor_plan', 'floor_space', 'target']].index[0]
df_all.drop(index=idx, inplace=True)
```

誤入力による外れ値を削除したので、再度ヒストグラムを表示します。

### 床面積のヒストグラム（外れ値消去）


```python
plt.figure(figsize=(14,8))
plt.hist(df_all[df_all.floor_space<=100].floor_space, bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_75_0.png)


分布は多峰性（山が複数ある）であることがわかります。これは違う間取りの分布を重ねた分布であるからだと推測できます。（つまり、例えば1Rだけの間取りでヒストグラムを見ると単峰性だが、違う間取りを重ねると多峰性になる）

### 築年数


```python
# 築0年のときは新築と記載されているので、築0年に置き換える。
df_all['how_old'] = df_all['how_old'].apply(lambda x: re.findall(' 築(\d+)年 ', x.replace('新築', '築0年'))[0]).astype('int')
```

### 築年数のヒストグラム


```python
plt.figure(figsize=(14,8))
plt.hist(df_all['how_old'], bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_80_0.png)


### 築年数のTOP10


```python
df_all.groupby('how_old').count().sort_values(by='address', ascending=False)[['address']].head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
    </tr>
    <tr>
      <th>how_old</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16453</td>
    </tr>
    <tr>
      <th>31</th>
      <td>7010</td>
    </tr>
    <tr>
      <th>30</th>
      <td>6768</td>
    </tr>
    <tr>
      <th>32</th>
      <td>6574</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6272</td>
    </tr>
    <tr>
      <th>33</th>
      <td>5581</td>
    </tr>
    <tr>
      <th>28</th>
      <td>5444</td>
    </tr>
    <tr>
      <th>27</th>
      <td>5279</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4527</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4418</td>
    </tr>
  </tbody>
</table>
</div>



新築、築30年前後、築15年前後の順に多いです。新築が多いのは当然だとして、築30年前後が築15年前後よりも多いのは「古い物件の家賃の安さ」よりも「適度な家賃と老朽度」の築15年前後の方が人気だからでしょう。

## 敷金・礼金
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

### 礼金のヒストグラム


```python
plt.figure(figsize=(14,8))
plt.hist(df_all['key_money'], bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_88_0.png)


礼金が最大の物件について見てみる


```python
df_all[df_all.key_money==df_all.key_money.max()][['how_old', 'floor_plan', 'floor_space', 'target', 'key_money']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>how_old</th>
      <th>floor_plan</th>
      <th>floor_space</th>
      <th>target</th>
      <th>key_money</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22390</th>
      <td>4</td>
      <td>1K</td>
      <td>28.4</td>
      <td>120000</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>



1Kの家賃12万の物件の礼金が家賃17ヶ月分は考えにくいので誤入力と思われます。


```python
idx = df_all[df_all.key_money==df_all.key_money.max()][['how_old', 'floor_plan', 'floor_space', 'target', 'key_money']].index[0]
df_all.drop(index=idx, inplace=True)
```

### 礼金のヒストグラム（外れ値削除）


```python
plt.figure(figsize=(14,8))
plt.hist(df_all['key_money'], bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_94_0.png)


### 敷金のヒストグラム


```python
plt.figure(figsize=(14,8))
plt.hist(df_all['security_deposit'], bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_96_0.png)



```python
df_all[df_all.security_deposit==df_all.security_deposit.max()][['how_old', 'floor_plan', 'floor_space', 'target', 'key_money', 'security_deposit']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>how_old</th>
      <th>floor_plan</th>
      <th>floor_space</th>
      <th>target</th>
      <th>key_money</th>
      <th>security_deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42337</th>
      <td>11</td>
      <td>1LDK</td>
      <td>40.02</td>
      <td>145000</td>
      <td>1.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>57490</th>
      <td>11</td>
      <td>1LDK</td>
      <td>40.02</td>
      <td>145000</td>
      <td>1.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>



これも敷金が11ヶ月は考えにくいですね。。


```python
idx = df_all[df_all.security_deposit==df_all.security_deposit.max()][['how_old', 'floor_plan', 'floor_space', 'target', 'key_money']].index
df_all.drop(index=idx, inplace=True)
```

## 更新料


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

### 更新料のヒストグラム


```python
plt.figure(figsize=(14,8))
plt.hist(df_all['renewal_fee'], bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_103_0.png)



```python
df_all[df_all.renewal_fee==df_all.renewal_fee.max()][['how_old', 'floor_plan', 'floor_space', 'target', 'key_money', 'security_deposit', 'renewal_fee']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>how_old</th>
      <th>floor_plan</th>
      <th>floor_space</th>
      <th>target</th>
      <th>key_money</th>
      <th>security_deposit</th>
      <th>renewal_fee</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30421</th>
      <td>10</td>
      <td>ワンルーム</td>
      <td>12.11</td>
      <td>65000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>



ワンルームの物件が更新費用が家賃16ヶ月分は考えにくいので削除します


```python
idx = df_all[df_all.renewal_fee==df_all.renewal_fee.max()][['how_old', 'floor_plan', 'floor_space', 'target', 'key_money']].index
df_all.drop(index=idx, inplace=True)
```

### 階数


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
df_all[df_all.floor==df_all.floor.max()][['how_old', 'floor_plan', 'floor_space', 'target', 'key_money', 'security_deposit', 'renewal_fee', 'floor']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>how_old</th>
      <th>floor_plan</th>
      <th>floor_space</th>
      <th>target</th>
      <th>key_money</th>
      <th>security_deposit</th>
      <th>renewal_fee</th>
      <th>floor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38283</th>
      <td>13</td>
      <td>1LDK</td>
      <td>33.39</td>
      <td>75000</td>
      <td>0.277778</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>301</td>
    </tr>
  </tbody>
</table>
</div>



301階は考えにくいので削除します。


```python
idx = df_all[df_all.floor==df_all.floor.max()][['how_old', 'floor_plan', 'floor_space', 'target', 'key_money']].index
df_all.drop(index=idx, inplace=True)
```

### 階数のヒストグラム


```python
plt.figure(figsize=(14,8))
plt.hist(df_all['floor'], bins=100)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_115_0.png)


`target`という変数を新たに作成したので不要なカラムは削除します


```python
df_all.drop(['rent', 'service_fee'], axis=1, inplace=True)
```

### 相関マップ


```python
mask = np.zeros_like(df_all.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (15,12))
sns.heatmap(df_all.corr().round(decimals=2), 
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


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_119_0.png)


### ヒートマップからわかること
- `auto_lock`と家賃の相関がやや大きい。ただし、`auto_lock`は築年数と大きな負の相関があり、これは`auto_lock`が導入され始めたのが比較的新しいためだと考えられる。  
したがって、家賃とオートロックは疑似相関（築年数を介した）の可能性が高い

## 偏相関係数の算出
家賃とオートロックの有無は相関があるように見えるが築年数という第三の変数を介した疑似相関である可能性があることに言及した。  
そこで偏相関係数を算出することで本当にそれらが疑似相関かどうかを定量的に見積もってみる。


```python
from pingouin import partial_corr

partial_corr(data=df_all, x='auto_lock', y='target', covar='how_old', method='pearson')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n</th>
      <th>r</th>
      <th>CI95%</th>
      <th>r2</th>
      <th>adj_r2</th>
      <th>p-val</th>
      <th>BF10</th>
      <th>power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pearson</th>
      <td>164679</td>
      <td>0.301206</td>
      <td>[0.3, 0.31]</td>
      <td>0.090725</td>
      <td>0.090714</td>
      <td>0.0</td>
      <td>inf</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



家賃とオートロックの相関係数は0.45だったが、第三の変数を築年数とした擬似相関は0.30となった。  
したがって、築年数の影響を除いた後の家賃とオートロックの関係はさほど強くないことがわかる。

### 住所


```python
df_all['address_town'] = df_all['address'].apply(lambda x: re.split('\d+', x)[0])
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
def extract_town_name(row):
    row = row.replace('-', '丁目')
    try:
        return [s for s in re.findall('(東京都.+区\D+)\d+丁目|(東京都.+市\D+)\d+丁目|(東京都.+郡\D+)\d+丁目', row)[0] if len(s)>0][0]
    except:
        return ""
```


```python
df_all['town'] = df_all['address'].apply(extract_town_name)
```

### 家賃が高い地域TOP30


```python
df_temp = df_all.groupby('town').target.agg(['mean', 'count'])
```


```python
df_temp[df_temp['count'] >= 10].sort_values(by='mean', ascending=False).head(30)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>town</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>東京都千代田区三番町</th>
      <td>333375.000000</td>
      <td>24</td>
    </tr>
    <tr>
      <th>東京都千代田区西神田</th>
      <td>303043.478261</td>
      <td>23</td>
    </tr>
    <tr>
      <th>東京都中央区晴海</th>
      <td>280552.183908</td>
      <td>87</td>
    </tr>
    <tr>
      <th>東京都港区東新橋</th>
      <td>250750.000000</td>
      <td>24</td>
    </tr>
    <tr>
      <th>東京都港区虎ノ門</th>
      <td>225257.575758</td>
      <td>33</td>
    </tr>
    <tr>
      <th>東京都渋谷区恵比寿西</th>
      <td>224439.393939</td>
      <td>33</td>
    </tr>
    <tr>
      <th>東京都中央区勝どき</th>
      <td>222157.746479</td>
      <td>355</td>
    </tr>
    <tr>
      <th>東京都江東区有明</th>
      <td>211911.764706</td>
      <td>34</td>
    </tr>
    <tr>
      <th>東京都渋谷区松濤</th>
      <td>211909.090909</td>
      <td>11</td>
    </tr>
    <tr>
      <th>東京都江東区豊洲</th>
      <td>209931.972789</td>
      <td>294</td>
    </tr>
    <tr>
      <th>東京都港区六本木</th>
      <td>209088.299320</td>
      <td>147</td>
    </tr>
    <tr>
      <th>東京都港区元麻布</th>
      <td>204975.000000</td>
      <td>40</td>
    </tr>
    <tr>
      <th>東京都渋谷区渋谷</th>
      <td>199958.333333</td>
      <td>60</td>
    </tr>
    <tr>
      <th>東京都中央区日本橋小舟町</th>
      <td>199900.000000</td>
      <td>20</td>
    </tr>
    <tr>
      <th>東京都港区南青山</th>
      <td>198618.131868</td>
      <td>182</td>
    </tr>
    <tr>
      <th>東京都江東区東雲</th>
      <td>197554.050193</td>
      <td>259</td>
    </tr>
    <tr>
      <th>東京都港区西新橋</th>
      <td>196265.151515</td>
      <td>66</td>
    </tr>
    <tr>
      <th>東京都渋谷区神宮前</th>
      <td>195959.016393</td>
      <td>61</td>
    </tr>
    <tr>
      <th>東京都新宿区払方町</th>
      <td>195842.105263</td>
      <td>19</td>
    </tr>
    <tr>
      <th>東京都中央区日本橋馬喰町</th>
      <td>192992.682927</td>
      <td>205</td>
    </tr>
    <tr>
      <th>東京都新宿区新小川町</th>
      <td>189780.000000</td>
      <td>50</td>
    </tr>
    <tr>
      <th>東京都渋谷区大山町</th>
      <td>188800.000000</td>
      <td>10</td>
    </tr>
    <tr>
      <th>東京都港区芝浦</th>
      <td>188309.727626</td>
      <td>257</td>
    </tr>
    <tr>
      <th>東京都港区浜松町</th>
      <td>181306.250000</td>
      <td>80</td>
    </tr>
    <tr>
      <th>東京都千代田区麹町</th>
      <td>179959.090909</td>
      <td>22</td>
    </tr>
    <tr>
      <th>東京都渋谷区恵比寿南</th>
      <td>179758.064516</td>
      <td>62</td>
    </tr>
    <tr>
      <th>東京都江東区辰巳</th>
      <td>178707.703704</td>
      <td>27</td>
    </tr>
    <tr>
      <th>東京都品川区上大崎</th>
      <td>176504.878049</td>
      <td>123</td>
    </tr>
    <tr>
      <th>東京都港区赤坂</th>
      <td>175818.892368</td>
      <td>511</td>
    </tr>
    <tr>
      <th>東京都港区台場</th>
      <td>175700.000000</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



### 家賃が安い地域TOP30


```python
df_temp[df_temp['count'] >= 10].sort_values(by='mean').head(30)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>town</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>東京都日野市程久保</th>
      <td>40216.571429</td>
      <td>175</td>
    </tr>
    <tr>
      <th>東京都八王子市滝山町</th>
      <td>42206.896552</td>
      <td>29</td>
    </tr>
    <tr>
      <th>東京都西東京市緑町</th>
      <td>43578.947368</td>
      <td>19</td>
    </tr>
    <tr>
      <th>東京都町田市忠生</th>
      <td>44971.264368</td>
      <td>87</td>
    </tr>
    <tr>
      <th>東京都武蔵村山市中藤</th>
      <td>44991.666667</td>
      <td>12</td>
    </tr>
    <tr>
      <th>東京都多摩市山王下</th>
      <td>45514.285714</td>
      <td>35</td>
    </tr>
    <tr>
      <th>東京都八王子市丹木町</th>
      <td>46466.019417</td>
      <td>103</td>
    </tr>
    <tr>
      <th>東京都八王子市寺町</th>
      <td>46678.571429</td>
      <td>14</td>
    </tr>
    <tr>
      <th>東京都多摩市馬引沢</th>
      <td>46726.027397</td>
      <td>146</td>
    </tr>
    <tr>
      <th>東京都八王子市追分町</th>
      <td>47133.333333</td>
      <td>30</td>
    </tr>
    <tr>
      <th>東京都八王子市中野上町</th>
      <td>47350.000000</td>
      <td>210</td>
    </tr>
    <tr>
      <th>東京都八王子市加住町</th>
      <td>47821.428571</td>
      <td>14</td>
    </tr>
    <tr>
      <th>東京都八王子市鑓水</th>
      <td>47966.666667</td>
      <td>24</td>
    </tr>
    <tr>
      <th>東京都八王子市暁町</th>
      <td>48626.943005</td>
      <td>193</td>
    </tr>
    <tr>
      <th>東京都多摩市中沢</th>
      <td>49220.338983</td>
      <td>59</td>
    </tr>
    <tr>
      <th>東京都八王子市中野山王</th>
      <td>49420.118343</td>
      <td>169</td>
    </tr>
    <tr>
      <th>東京都町田市東玉川学園</th>
      <td>49458.333333</td>
      <td>24</td>
    </tr>
    <tr>
      <th>東京都八王子市寺田町</th>
      <td>49586.666667</td>
      <td>15</td>
    </tr>
    <tr>
      <th>東京都あきる野市二宮東</th>
      <td>49947.368421</td>
      <td>19</td>
    </tr>
    <tr>
      <th>東京都八王子市万町</th>
      <td>50000.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>東京都多摩市南野</th>
      <td>50270.833333</td>
      <td>48</td>
    </tr>
    <tr>
      <th>東京都八王子市四谷町</th>
      <td>50615.384615</td>
      <td>13</td>
    </tr>
    <tr>
      <th>東京都青梅市千ヶ瀬町</th>
      <td>50738.095238</td>
      <td>42</td>
    </tr>
    <tr>
      <th>東京都東村山市多摩湖町</th>
      <td>50992.307692</td>
      <td>39</td>
    </tr>
    <tr>
      <th>東京都八王子市大塚</th>
      <td>51386.106870</td>
      <td>262</td>
    </tr>
    <tr>
      <th>東京都小平市たかの台</th>
      <td>51500.000000</td>
      <td>11</td>
    </tr>
    <tr>
      <th>東京都八王子市左入町</th>
      <td>51500.000000</td>
      <td>13</td>
    </tr>
    <tr>
      <th>東京都あきる野市小川東</th>
      <td>51571.428571</td>
      <td>14</td>
    </tr>
    <tr>
      <th>東京都八王子市小比企町</th>
      <td>51590.909091</td>
      <td>33</td>
    </tr>
    <tr>
      <th>東京都青梅市東青梅</th>
      <td>51696.629213</td>
      <td>89</td>
    </tr>
  </tbody>
</table>
</div>



## その他の特徴量のヒストグラム


```python
fig = df_all.hist(bins=50, figsize=(20,90), layout=(25, 1), xlabelsize=15, ylabelsize=15)
[x.title.set_size(25) for x in fig.ravel()]
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_136_0.png)


## 特別区の地図上に平均家賃価格を可視化


```python
import geopandas as gpd

# get from https://github.com/dataofjapan/land
df_tokyomap = gpd.read_file('./data/tokyo.geojson')
df_toshin = df_tokyomap[df_tokyomap['area_en'] == 'Tokubu'].copy()
df_toshin['mean'] = [df_all.groupby('city').mean()[['target']].loc[s].target for s in df_toshin.ward_ja]
df_toshin['coords'] = df_toshin.geometry.apply(lambda x: x.representative_point().coords[:][0])
```


```python
df_toshin.plot(column='mean', figsize=(20,12), legend=True, legend_kwds={'label': "Mean rent"}, cmap='viridis')
for idx, row in df_toshin.iterrows():
    plt.annotate(s=row['ward_en'], xy=row['coords'],
                 horizontalalignment='center', color='white')
plt.title('Average rent for each city', fontsize=25)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_139_0.png)


港区、中央区を中心として放射線状に価格が分布していることがわかります。

### APIで経度・緯度を取得し、地区の平均家賃価格をプロット


```python
import ast

# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter

# geolocator = Nominatim()
# geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# location_dict = {}
# for i,s in enumerate(df_all.town.unique()):
#     print(i)
#     if s:
#         loc = geocode(s)
#         if loc:
#             location_dict[s] = loc[1]
# pd.DataFrame.from_dict({'location': location_dict}).to_csv('data/location.csv')

df_loc = pd.read_csv('data/location.csv', index_col=0, converters={"location": ast.literal_eval})
df_loc['latitude'] = [s[0] for s in df_loc.location]
df_loc['longitude'] = [s[1] for s in df_loc.location]

df_loc_target = pd.merge(df_all[df_all.town.isin(df_loc.index)].groupby('town').mean()[['time_to_go_nearest_station']], df_loc ,left_index=True, right_index=True)

df_toshin.plot(column='mean', figsize=(20,12), legend=True, legend_kwds={'label': "Mean rent"}, cmap='viridis')
for idx, row in df_toshin.iterrows():
    plt.annotate(s=row['ward_en'], xy=row['coords'],
                 horizontalalignment='center', color='white')
plt.scatter(df_loc_target[(df_loc_target.latitude >=35) & (df_loc_target.longitude >= 139.6)].longitude,
            df_loc_target[(df_loc_target.latitude >=35) & (df_loc_target.longitude >= 139.6)].latitude,
            s=df_loc_target.time_to_go_nearest_station*20, alpha=0.6, c='red')
plt.title('Average rent for each city', fontsize=25)
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_142_0.png)


この図からあんまり得られる情報はありません。こんなことも出来るんだ程度に見てください。


```python
numeric_col = [s for s in df_all.columns if df_all[s].dtype != 'object']
object_col = ['city', 'floor_plan', 'first_near_station', 'second_near_station', 'third_near_station', 'status', 'address', 'address_town', 'name', 'url', 'structure']
```


```python
df = df_all[numeric_col + object_col]
```

目的変数の対数変換


```python
df['target'] = np.log1p(df['target'].copy())
```


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
    [500]	training's rmse: 0.0620642	valid_1's rmse: 0.0985041
    [1000]	training's rmse: 0.0487933	valid_1's rmse: 0.0957398
    [1500]	training's rmse: 0.0401117	valid_1's rmse: 0.0944433
    [2000]	training's rmse: 0.0343744	valid_1's rmse: 0.0938105
    [2500]	training's rmse: 0.0297608	valid_1's rmse: 0.0934359
    [3000]	training's rmse: 0.0260618	valid_1's rmse: 0.0932451
    [3500]	training's rmse: 0.023066	valid_1's rmse: 0.093079
    [4000]	training's rmse: 0.0205726	valid_1's rmse: 0.0929632
    [4500]	training's rmse: 0.0185568	valid_1's rmse: 0.0928923
    [5000]	training's rmse: 0.0168284	valid_1's rmse: 0.0928516
    [5500]	training's rmse: 0.0154795	valid_1's rmse: 0.0928408
    [6000]	training's rmse: 0.0142143	valid_1's rmse: 0.0928176
    [6500]	training's rmse: 0.0132527	valid_1's rmse: 0.0928068
    [7000]	training's rmse: 0.0123708	valid_1's rmse: 0.092807
    Early stopping, best iteration is:
    [6865]	training's rmse: 0.0125814	valid_1's rmse: 0.0927934
    fold 1
    Training until validation scores don't improve for 500 rounds
    [500]	training's rmse: 0.0623229	valid_1's rmse: 0.097126
    [1000]	training's rmse: 0.0493847	valid_1's rmse: 0.0944533
    [1500]	training's rmse: 0.0411358	valid_1's rmse: 0.0933576
    [2000]	training's rmse: 0.0350585	valid_1's rmse: 0.0926453
    [2500]	training's rmse: 0.0304515	valid_1's rmse: 0.0922679
    [3000]	training's rmse: 0.0266654	valid_1's rmse: 0.0920523
    [3500]	training's rmse: 0.0236033	valid_1's rmse: 0.0918693
    [4000]	training's rmse: 0.021205	valid_1's rmse: 0.0917438
    [4500]	training's rmse: 0.0190847	valid_1's rmse: 0.0916698
    [5000]	training's rmse: 0.01743	valid_1's rmse: 0.0916113
    [5500]	training's rmse: 0.015964	valid_1's rmse: 0.0915749
    [6000]	training's rmse: 0.0148459	valid_1's rmse: 0.0915459
    [6500]	training's rmse: 0.0137299	valid_1's rmse: 0.0915021
    [7000]	training's rmse: 0.0127857	valid_1's rmse: 0.0914938
    [7500]	training's rmse: 0.0119904	valid_1's rmse: 0.0914903
    [8000]	training's rmse: 0.0113017	valid_1's rmse: 0.0914776
    [8500]	training's rmse: 0.0107149	valid_1's rmse: 0.0914776
    [9000]	training's rmse: 0.0101918	valid_1's rmse: 0.0914608
    [9500]	training's rmse: 0.00978279	valid_1's rmse: 0.0914645
    Early stopping, best iteration is:
    [9132]	training's rmse: 0.0100941	valid_1's rmse: 0.0914572
    fold 2
    Training until validation scores don't improve for 500 rounds
    [500]	training's rmse: 0.0630023	valid_1's rmse: 0.0977646
    [1000]	training's rmse: 0.0493929	valid_1's rmse: 0.0949977
    [1500]	training's rmse: 0.0406841	valid_1's rmse: 0.0937988
    [2000]	training's rmse: 0.0349311	valid_1's rmse: 0.0932718
    [2500]	training's rmse: 0.0301731	valid_1's rmse: 0.0929056
    [3000]	training's rmse: 0.0264926	valid_1's rmse: 0.0926949
    [3500]	training's rmse: 0.0234339	valid_1's rmse: 0.0925585
    [4000]	training's rmse: 0.0209832	valid_1's rmse: 0.0924572
    [4500]	training's rmse: 0.0189538	valid_1's rmse: 0.0923834
    [5000]	training's rmse: 0.01722	valid_1's rmse: 0.0923476
    [5500]	training's rmse: 0.0158124	valid_1's rmse: 0.0923265
    [6000]	training's rmse: 0.0145178	valid_1's rmse: 0.092309
    [6500]	training's rmse: 0.0135356	valid_1's rmse: 0.0922969
    [7000]	training's rmse: 0.0126584	valid_1's rmse: 0.0923015
    Early stopping, best iteration is:
    [6703]	training's rmse: 0.0131689	valid_1's rmse: 0.0922942
    RMSE 10735.980383386666円
    CPU times: user 57min 49s, sys: 37.7 s, total: 58min 27s
    Wall time: 5min 17s


## 変数重要度の可視化


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


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_155_0.png)


## 精度評価


```python
pred_list = []
for model in models:
    pred_list.append(model.predict(X_test.drop(['url', 'name'], axis=1)))
```


```python
y_pred = np.zeros_like(pred_list[0])
for arr in pred_list:
    y_pred += arr
y_pred /= len(pred_list)
```


```python
plt.figure(figsize=(12, 12))
plt.scatter(np.expm1(y_pred), np.expm1(y_test), alpha=0.3)
plt.tight_layout()
plt.show()
```


![png](%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_files/%E4%B8%8D%E5%8B%95%E7%94%A3%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_159_0.png)



```python
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

## 相対誤差


```python
mean_absolute_percentage_error(np.expm1(y_test), np.expm1(y_pred))
```




    5.933115251734478




```python
def calc_sc(x):
    return np.round_(50+10*(x-np.average(x))/np.std(x))
```


```python
if model_save:
    for idx, model in enumerate(models):
        model.booster_.save_model(f'./data/lgb_regressor_{idx}.txt')
    X_train = X_train.reset_index(drop=True).join(df_fold_all)
    X_train['target'] = np.expm1(y_train)
    X_train['predict'] = np.expm1(oof)
    X_train['mape'] = X_train.apply(lambda x: (x['predict']-x['target'])/x['target'], axis=1)
    X_train['sc'] = calc_sc(X_train['mape'])
    X_train.to_csv('./data/x_train.csv', index=False)
```


```python
# !rm -r 不動産価格予測_files
# !jupyter nbconvert --to markdown 不動産価格予測.ipynb
# !mv 不動産価格予測.md README.md
```
