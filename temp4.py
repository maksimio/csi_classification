from metawifi import WifiDf, WifiLearn
import pandas as pd


wd = WifiDf('./csi/homelocation/three place').set_type('abs').diff()
wl = WifiLearn(*wd.prep_csi())
data = wl.fit_regression()

data = (data / 1).round() * 1
df = pd.DataFrame()
df['real'] = wl.y_test
df['predict'] = data
df['equal'] = df['real'] == df['predict']
print(df)
print(df['equal'].sum() / df.shape[0] * 100)
