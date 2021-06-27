import wifi_log.logreader as lr


lr.Log.run_lib('./dll/readcsi.dll')
log1 = lr.Log('./csi/homelocation/five place/test/bathroom4.dat').read()
print(len(log1))
log2 = lr.Log('./csi/use_in_paper/2_objects/train/D=2020-01-02_T=21-34-18--air.dat').read()
print(len(log2))


log = log1 + log2
print(len(log))



a = 5