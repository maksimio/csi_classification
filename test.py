import pandas as pd
from metawifi import WifiDf, WifiLearn
from matplotlib import pyplot as plt

wd = WifiDf('./csi/homelocation/two place').set_type('abs')
wl = WifiLearn(*wd.prep_csi()).fit_cnn().print()