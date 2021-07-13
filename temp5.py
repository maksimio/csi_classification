from metawifi import WifiDf, WifiLearn
import pandas as pd


wd = WifiDf('./csi/homelocation/three place')

df = wd.df.groupby(['type', 'category']).count()
print(df)


wd = WifiDf('./csi/homelocation/two place')

df = wd.df.groupby(['type', 'category']).count()
print(df)

