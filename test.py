from wifi_log.log import Log
from time import time

# Log.run_lib('./dll/readcsi.dll')
start = time()

for i in range(100):
    log2 = Log('./csi/use_in_paper/2_objects/train/D=2020-01-02_T=21-34-18--air.dat').read()

print(time() - start)
# print(log2.raw[0]['csi_raw'])
a = 5



# log2 = Log('./csi/use_in_paper/2_objects/train/D=2020-01-02_T=21-34-18--air.dat').read()