import pandas as pd
from metawifi import MetaWifi, MetaLearn
from matplotlib import pyplot as plt


mf = MetaWifi('./csi/use_in_paper/4_objects').set_type('phase').unjump().diff(5).smooth(25, 'hamming').view()
lf = MetaLearn(*mf.prep_csi())
res = lf.fit_classic()
print(res)
