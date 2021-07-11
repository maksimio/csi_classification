from metawifi import WifiDf, WifiLearn


wd = WifiDf('./csi/homelocation/three place').set_type('abs')
wl = WifiLearn(*wd.prep_csi()).fit_classic()
