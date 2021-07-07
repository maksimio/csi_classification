from metawifi import WifiDf, WifiLearn


wd = WifiDf('./csi/use_in_paper/2_objects').set_type('abs')
wd.df_csi_abs = wd.df_csi_abs / 400
wl = WifiLearn(*wd.prep_csi()).fit_ffnn()

print('-----------------------------------------------------')

wd.restore_csi().diff()
wl = WifiLearn(*wd.prep_csi()).fit_ffnn()