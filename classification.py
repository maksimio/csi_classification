from metawifi.df import wifilearn
from metawifi import WifiDf, WifiLearn
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import seaborn as sns


wd = WifiDf('./csi/use_in_paper/2_objects').set_type('abs')
# wd = WifiDf('./csi/homelocation/three place').set_type('abs')
x, y, z, a = wd.prep_featurespace()
df = pd.DataFrame(x)
df['target'] = y
# print(x)
sns.lmplot(x='skew_1', y='kurt_1', data=df, hue='target', fit_reg=False)
sns.lmplot(x='skew_2', y='kurt_2', data=df, hue='target', fit_reg=False)
sns.lmplot(x='skew_3', y='kurt_3', data=df, hue='target', fit_reg=False)
sns.lmplot(x='std_1', y='mu42_1', data=df, hue='target', fit_reg=False)

plt.show()
exit()
wl = WifiLearn(*wd.prep_csi()).fit_classic().print()


# https://habr.com/ru/company/ods/blog/325422/