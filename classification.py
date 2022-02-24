from metawifi.df import wifilearn
from metawifi import WifiDf, WifiLearn
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import seaborn as sns


wd = WifiDf('./csi/use_in_paper/2_objects').set_type('abs')
x, y, z, a = wd.prep_featurespace()
df = pd.DataFrame(x)
df['target'] = y


wl = WifiLearn(*wd.prep_csi()).fit_classic().print()


# https://habr.com/ru/company/ods/blog/325422/