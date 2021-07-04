import pandas as pd
from metawifi import MetaWifi
from matplotlib import pyplot as plt


mf = MetaWifi('./csi/use_in_paper/4_objects').set_type('phase')

mf.unjump().diff().smooth(win_type='hamming')

mf.set_type('abs')
mf.smooth(10, 'hamming')

mf.df_csi_phase.head(20).T.plot()
mf.df_csi_abs.head(20).T.plot()
plt.show()