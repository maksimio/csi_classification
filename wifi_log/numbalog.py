import ctypes
import numpy as np
from struct import unpack
from os import path
from numba import njit, jit


@njit(cache=True)
def signbit_convert(data):
    BITS_PER_SYMBOL = 10

    if data & (1 << (BITS_PER_SYMBOL - 1)):
        data -= (1 << BITS_PER_SYMBOL)
    return data


@njit(cache=True)
def _read_csi_native(local_h, nr, nc, num_tones):
    csi_re = [[0 for i in range(num_tones)] for k in range(nr * nc)]
    csi_im = [[0 for i in range(num_tones)] for k in range(nr * nc)]

    BITS_PER_BYTE = 8
    BITS_PER_SYMBOL = 10
    bits_left = 16

    bitmask = (1 << BITS_PER_SYMBOL) - 1
    h_data = local_h[0] + (local_h[1] << BITS_PER_BYTE)
    idx = 2
    current_data = h_data & ((1 << 16) - 1)

    for k in range(num_tones):
        for nc_idx in range(nc):
            for nr_idx in range(nr):
                if bits_left - BITS_PER_SYMBOL < 0:
                    h_data = local_h[idx] + (local_h[idx + 1] << BITS_PER_BYTE)
                    idx += 2
                    current_data += h_data << bits_left
                    bits_left += 16
                
                imag = current_data & bitmask
                bits_left -= BITS_PER_SYMBOL
                current_data = current_data >> BITS_PER_SYMBOL

                if bits_left - BITS_PER_SYMBOL < 0:
                    h_data = local_h[idx] + (local_h[idx + 1] << BITS_PER_BYTE)
                    idx += 2
                    current_data += h_data << bits_left
                    bits_left += 16

                real = current_data & bitmask
                bits_left -= BITS_PER_SYMBOL
                current_data = current_data >> BITS_PER_SYMBOL

                csi_re[nr_idx + nc_idx * 2][k] = signbit_convert(real)
                csi_im[nr_idx + nc_idx * 2][k] = signbit_convert(imag)

    return csi_re, csi_im


class Log:
    def _read_csi(self, csi_buf: list, nr: int, nc: int, num_tones: int) -> dict:
        csi_re, csi_im = _read_csi_native(csi_buf, nr, nc, num_tones)
        a = 5
        return {
            'csi_on_path_1': np.array(csi_re[0][:]) + 1j * np.array(csi_im[0][:]), 
            'csi_on_path_2': np.array(csi_re[1][:]) + 1j * np.array(csi_im[1][:]), 
            'csi_on_path_3': np.array(csi_re[2][:]) + 1j * np.array(csi_im[2][:]), 
            'csi_on_path_4': np.array(csi_re[3][:]) + 1j * np.array(csi_im[3][:])}


    def __init__(self, path: str) -> None:
        self.path = path
        self.raw = []

    
    def read(self):
        with open(self.path, 'rb') as f:
            len_file = path.getsize(self.path)
            cur = 0

            while cur < len_file:
                csi_matrix = {}
                csi_matrix['field_len'] = unpack('>H', f.read(2))[0] # field_len doesn`t use
                csi_matrix['timestamp'], csi_matrix['csi_len'], csi_matrix['tx_channel'], csi_matrix['err_info'], csi_matrix['noise_floor'], csi_matrix['rate'],  csi_matrix['bandWitdh'], csi_matrix['num_tones'],  csi_matrix['nr'], csi_matrix['nc'], csi_matrix['rssi0'], csi_matrix['rssi1'], csi_matrix['rssi2'], csi_matrix['rssi3'], csi_matrix['payload_len'] = unpack('>QHHBBBBBBBBBBBH', f.read(25))
               
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