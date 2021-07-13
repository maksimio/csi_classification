from metawifi import WifiDf, WifiLearn
from matplotlib import pyplot as plt
import statsmodels.api as sm

wd = WifiDf('./csi/use_in_paper/4_objects').set_type('abs')
# wd = WifiDf('./csi/homelocation/three place').set_type('abs')


df_zscore = (wd.df_csi_abs - wd.df_csi_abs.mean())/wd.df_csi_abs.std()
sm.qqplot(df_zscore[20], line='45')
sm.qqplot(df_zscore[30], line='45')
sm.qqplot(df_zscore[80], line='45')
sm.qqplot(df_zscore[150], line='45')
plt.show()


exit()
wl = WifiLearn(*wd.prep_csi()).fit_classic().print()
