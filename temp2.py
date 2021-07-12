from metawifi import WifiDf, WifiLearn

wd = WifiDf('./csi/homelocation/three place').set_type('abs')
wd.smooth(win='hamming').diff()
# wd.df_csi_abs = wd.df_csi_abs.abs()
# wd.view()

wl = WifiLearn(*wd.prep_csi()).fit_cnn(1000, 200)