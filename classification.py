from metawifi import WifiDf, WifiLearn
import pandas as pd
from matplotlib import pyplot as plt
import scipy


wd = WifiDf('./csi/use_in_paper/2_objects').set_type('abs')

x_train, y_train, x_test, y_test = wd.prep_csi()
df_train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
print(df_train)
df_train_air = df_train[df_train['category'] == 'bottle']
df_test = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)
df_test_air = df_test[df_test['category'] == 'bottle']

df_train_air.boxplot([i for i in range(224)], color='green')
df_test_air.boxplot([i for i in range(224)], color='violet')
df_train_air.hist(30)
fft = scipy.fft(df_test_air[42])

plt.plot(fft)
plt.show()
