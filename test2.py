import pandas as pd
from metawifi import MetaWifi
from matplotlib import pyplot as plt

mf = MetaWifi('./csi/use_in_paper/2_objects')

print(mf.df.groupby('type', axis=0).count())