import ctypes
import numpy as np
from struct import unpack
from os import path


class Reader:
    def __init__(self, lib_path, nr:int=2, nc:int=2, num_tones:int=56) -> None:
        self.lib = ctypes.CDLL(lib_path)
        self.lib.read_csi.restype = None
        self.lib.read_csi.argtypes = (
        ctypes.POINTER(ctypes.c_ubyte), 
        ctypes.POINTER(ctypes.c_int),     
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int))
        self.nr, self.nc, self.num_tones = nr, nc, num_tones
        self.csi_re = [[0 for i in range(self.num_tones)] for k in range(self.nr * self.nc)]
        self.csi_im = [[0 for i in range(self.num_tones)] for k in range(self.nr * self.nc)]
        for i in range(self.nr * self.nc):
            self.csi_re[i] = (ctypes.c_int * len(self.csi_re[i]))(*self.csi_re[i])
            self.csi_im[i] = (ctypes.c_int * len(self.csi_im[i]))(*self.csi_im[i])


    def _read_csi(self, csi_buf: list, nr: int, nc: int, num_tones: int) -> dict:
        if nr != self.nr or nc != self.nc or num_tones != self.num_tones:
            raise TypeError('Error: nr != self or nc != self or num_tones != self')

        csi_buf = (ctypes.c_ubyte * len(csi_buf))(*csi_buf)
        self.lib.read_csi(csi_buf, self.csi_re[0], self.csi_re[1], self.csi_re[2], self.csi_re[3], self.csi_im[0], self.csi_im[1], self.csi_im[2], self.csi_im[3])

        return {
            'csi_on_path_1': np.array(self.csi_re[0][:]) + 1j * np.array(self.csi_im[0][:]), 
            'csi_on_path_2': np.array(self.csi_re[1][:]) + 1j * np.array(self.csi_im[1][:]), 
            'csi_on_path_3': np.array(self.csi_re[2][:]) + 1j * np.array(self.csi_im[2][:]), 
            'csi_on_path_4': np.array(self.csi_re[3][:]) + 1j * np.array(self.csi_im[3][:])}


class Log:
    def __init__(self, reader: Reader, path: str, save_payload: bool=False) -> None:
        self.path = path
        self.save_payload = save_payload
        self.reader = reader
        self.log = None
    
    def read(self):
        with open(self.path, 'rb') as f:
            len_file = path.getsize(self.path)
            ret, cur = [], 0

            while cur < (len_file - 4):
                field_len = unpack('>H', f.read(2))[0]
                cur += 2
                if cur + field_len > len_file:
                    break

                csi_matrix = {}
                csi_matrix['timestamp'], csi_matrix['csi_len'], csi_matrix['tx_channel'], csi_matrix['err_info'], csi_matrix['noise_floor'], csi_matrix['rate'],  csi_matrix['bandWitdh'], csi_matrix['num_tones'],  csi_matrix['nr'], csi_matrix['nc'], csi_matrix['rssi'], csi_matrix['rssi1'], csi_matrix['rssi2'], csi_matrix['rssi3'], csi_matrix['payload_len'] = unpack('>QHHBBBBBBBBBBBH', f.read(25))
                cur += 25

                if csi_matrix['csi_len']:
                    # This condition saves a decent amount of time
                    csi_buf = unpack('B' * csi_matrix['csi_len'], f.read(csi_matrix['csi_len']))
                    csi_matrix.update(self.reader._read_csi(csi_buf, csi_matrix['nr'], csi_matrix['nc'], csi_matrix['num_tones']))
                    cur += csi_matrix['csi_len']

                if csi_matrix['payload_len']:
                    payload = unpack('B' * csi_matrix['payload_len'], f.read(csi_matrix['payload_len']))
                    cur += csi_matrix['payload_len']
                    if self.save_payload:
                        csi_matrix['payload'] = payload
                    else:
                        csi_matrix['payload'] = None
                else:
                    csi_matrix['payload'] = None

                if (cur + 420) > len_file:
                    break
                if csi_matrix['payload_len'] >= 1000 and csi_matrix['csi_len']:
                    csi_matrix['is_special'] = True
                else:
                    csi_matrix['is_special'] = False
                
                ret.append(csi_matrix)
                
        self.data = ret
