from metawifi.log.logcombiner import LogCombiner

lc = LogCombiner(LogCombiner.train_test('./csi/homelocation/five place'))
lc.combine()


a = 5