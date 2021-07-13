from metawifi import WifiDf, WifiLearn


wd = WifiDf('./csi/homelocation/three place').set_type('abs').diff()
df = WifiLearn(*wd.prep_csi()).fit_logistic_test()
print(df)
df.to_csv('temp.csv', index=False)
