import pandas as pd
from metawifi import WifiDf, WifiLearn
from matplotlib import pyplot as plt

wd = WifiDf('./csi/use_in_paper/2_objects').set_type('abs')
wl = WifiLearn(*wd.prep_rssi()).fit_classic().print()

wl = WifiLearn(*wd.prep_csi()).fit_classic().print()
wd.scale(0.01)
wl = WifiLearn(*wd.prep_csi()).fit_classic().print()