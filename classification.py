from metawifi.df import wifilearn
import pandas as pd
from metawifi import WifiDf, WifiLearn
from matplotlib import pyplot as plt

wd = WifiDf('./csi/use_in_paper/2_objects').set_type('abs').smooth(5, 'hamming').diff()
wl = WifiLearn(*wd.prep_rssi()).fit_classic().print()
wl = WifiLearn(*wd.prep_csi()).fit_ffnn(10, 5)
# wl.fit_classic()