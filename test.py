import wifi_log.logreader as lr
 

reader = lr.Reader('./dll/readcsi.dll')

log = lr.Log(reader, './csi/use_in_paper/2_objects/train/D=2020-01-02_T=21-09-35--air.dat')
log.read()

data = log.data
print(len(log.data))
a = 5