import pandas as pd
from metawifi import MetaWifi

mf = MetaWifi('./csi/use_in_paper/4_objects')
print(mf.df_csi_complex)
print(mf.df)
print(mf.df_csi_abs)
print(mf.df_csi_phase)