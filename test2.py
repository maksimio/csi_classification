import pandas as pd
from metawifi import MetaWifi
from matplotlib import pyplot as plt


mf = MetaWifi('./csi/use_in_paper/2_objects').set_type('phase')
mf.unjump().smooth(win='hamming')

mf.df_csi_phase.head(14).T.plot()
plt.show()