from time import time
dll_way = True

if dll_way:
    from metawifi.log_old import Log as LogReader
    LogReader.run_lib('./dll/readcsi.dll')
else:
    from metawifi.log.logreader import LogReader


start = time()

for i in range(50):
    log2 = LogReader('./csi/homelocation/five place/train/bathroom1.dat').read()
print(time() - start)
print(log2.raw[0]['csi'][3])
