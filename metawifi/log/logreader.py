from ..watcher import Watcher as W
from numpy import zeros
from struct import unpack
from os import path
from numba import njit


@njit(cache=True)
def __signbit_convert(data: int) -> int:
    if data & 512:
        data -= 1024
    return data


@njit(cache=True)
def _read_csi_native(local_h: int, nr: int, nc: int, num_tones: int) -> list:
    csi_re = zeros((nr * nc, num_tones))
    csi_im = zeros((nr * nc, num_tones))

    BITS_PER_BYTE = 8
    BITS_PER_SYMBOL = 10
    bits_left = 16

    h_data = local_h[0] + (local_h[1] << BITS_PER_BYTE)
    current_data = h_data & 65535
    idx = 2

    for k in range(num_tones):
        for nc_idx in range(nc):
            for nr_idx in range(nr):
                if bits_left < BITS_PER_SYMBOL:
                    h_data = local_h[idx] + (local_h[idx + 1] << BITS_PER_BYTE)
                    idx += 2
                    current_data += h_data << bits_left
                    bits_left += 16
                
                imag = current_data & 1023
                bits_left -= BITS_PER_SYMBOL
                current_data = current_data >> BITS_PER_SYMBOL

                if bits_left < BITS_PER_SYMBOL:
                    h_data = local_h[idx] + (local_h[idx + 1] << BITS_PER_BYTE)
                    idx += 2
                    current_data += h_data << bits_left
                    bits_left += 16

                real = current_data & 1023
                bits_left -= BITS_PER_SYMBOL
                current_data = current_data >> BITS_PER_SYMBOL

                csi_re[nr_idx + nc_idx * 2, k] = __signbit_convert(real)
                csi_im[nr_idx + nc_idx * 2, k] = __signbit_convert(imag)

    return csi_re, csi_im


class LogReader:
    def __read_csi(self, csi_buf: list, nr: int, nc: int, num_tones: int) -> dict:
        csi_re, csi_im = _read_csi_native(csi_buf, nr, nc, num_tones)
        return [csi_re[i] + 1j * csi_im[i] for i in range(nr * nc)]


    def __init__(self, path: str) -> None:
        self.path = path
        self.raw = []

    @W.stopwatch
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
                    csi_matrix['csi'] = self.__read_csi(buf, csi_matrix['nr'], csi_matrix['nc'], csi_matrix['num_tones'])
                else:
                    csi_matrix['csi'] = [] 
                               
                csi_matrix['payload'] = unpack('B' * csi_matrix['payload_len'], f.read(csi_matrix['payload_len']))
                cur += 27 + csi_matrix['csi_len'] + csi_matrix['payload_len']
                
                self.raw.append(csi_matrix)

        return self

    
    def add(self, name: str=None, value=None):
        if name == None:
            name = 'path'
        if value == None:
            value = self.path

        for dictionary in self.raw:
            dictionary.update({ name: value })
        
        return self

    
    def __getitem__(self, key: int) -> dict:
        return self.raw[key]


    def __add__(self, other: object):
        self.raw += other.raw
        return self


    def __len__(self) -> int:
        return len(self.raw)