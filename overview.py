from metawifi import WifiDf, WifiLearn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

wd = WifiDf('./csi/use_in_paper/2_objects').set_type('abs')
x_train, y_train, x_test, y_test = wd.prep_csi()
df_train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)

corrmat = df_train.corr()
# ax, f = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True, xticklabels=[i for i in range(224)])
# plt.show()
print(corrmat)

k = 10
cols = corrmat.nlargest(k, 'category')['category'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()