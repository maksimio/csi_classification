import pandas as pd
from metawifi import MetaWifi
from matplotlib import pyplot as plt

mf = MetaWifi('./csi/use_in_paper/4_objects')
mf.set_type('abs').smooth(20, 'hamming')
print(mf.df_csi_complex)
mf.df_csi_abs.head(20).T.plot()
plt.show()