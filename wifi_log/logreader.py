import ctypes
import numpy as np
from struct import unpack
from os import path


class Log:
    lib = None
    csi_re, csi_im  = None, None
    csi_im = []
    nr, nc, num_tones = None, None, None
    lib_path = None


    def _read_csi(self, csi_buf: list, nr: int, nc: int, num_tones: int) -> dict:
        if nr != Log.nr or nc != Log.nc or num_tones != Log.num_tones:
            raise ValueError('Error: nr != self or nc != self or num_tones != self')

        csi_buf = (ctypes.c_ubyte * len(csi_buf))(*csi_buf)
        Log.lib.read_csi(csi_buf, Log.csi_re[0], Log.csi_re[1], Log.csi_re[2], Log.csi_re[3], Log.csi_im[0], Log.csi_im[1], Log.csi_im[2], Log.csi_im[3])

        return {
            'csi_on_path_1': np.array(Log.csi_re[0][:]) + 1j * np.array(Log.csi_im[0][:]), 
            'csi_on_path_2': np.array(Log.csi_re[1][:]) + 1j * np.array(Log.csi_im[1][:]), 
            'csi_on_path_3': np.array(Log.csi_re[2][:]) + 1j * np.array(Log.csi_im[2][:]), 
            'csi_on_path_4': np.array(Log.csi_re[3][:]) + 1j * np.array(Log.csi_im[3][:])}
            

    def run_lib(lib_path: str, nr: int=2, nc: int=2, num_tones: int=56) -> None:
        Log.lib_path = lib_path
        Log.lib = ctypes.CDLL(lib_path)
        Log.lib.read_csi.restype = None
        Log.lib.read_csi.argtypes = (
        ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_int),     
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int))
        Log.nr, Log.nc, Log.num_tones = nr, nc, num_tones
        Log.csi_re = [[0 for i in range(Log.num_tones)] for k in range(Log.nr * Log.nc)]
        Log.csi_im = [[0 for i in range(Log.num_tones)] for k in range(Log.nr * Log.nc)]
        for i in range(Log.nr * Log.nc):
            Log.csi_re[i] = (ctypes.c_int * len(Log.csi_re[i]))(*Log.csi_re[i])
            Log.csi_im[i] = (ctypes.c_int * len(Log.csi_im[i]))(*Log.csi_im[i])
        

    def __init__(self, path: str) -> None:
        self.path = path
        self.raw = None

        if Log.lib == None:
            raise ValueError('Call run_lib static method!')

    
    def read(self):
        with open(self.path, 'rb') as f:
            len_file = path.getsize(self.path)
            self.raw, cur = [], 0

            while cur < len_file:
                csi_matrix = {}
                csi_matrix['field_len'] = unpack('>H', f.read(2))[0] # field_len doesn`t use
                csi_matrix['timestamp'], csi_matrix['csi_len'], csi_matrix['tx_channel'], csi_matrix['err_info'], csi_matrix['noise_floor'], csi_matrix['rate'],  csi_matrix['bandWitdh'], csi_matrix['num_tones'],  csi_matrix['nr'], csi_matrix['nc'], csi_matrix['rssi'], csi_matrix['rssi1'], csi_matrix['rssi2'], csi_matrix['rssi3'], csi_matrix['payload_len'] = unpack('>QHHBBBBBBBBBBBH', f.read(25))
               
                if csi_matrix['csi_len']:
                    buf = unpack('B' * csi_matrix['csi_len'], f.read(csi_matrix['csi_len']))
                    csi_matrix['csi_raw'] = self._read_csi(buf, csi_matrix['nr'], csi_matrix['nc'], csi_matrix['num_tones'])
                
                csi_matrix['payload'] = unpack('B' * csi_matrix['payload_len'], f.read(csi_matrix['payload_len']))
                cur += 27 + csi_matrix['csi_len'] + csi_matrix['payload_len']
                
                self.raw.append(csi_matrix)

        return self

    
    def __getitem__(self, key):
        return self.raw[key]

    def __add__(self, other):
        self.raw += other.raw
        return self

    def __len__(self):
        return len(self.raw)