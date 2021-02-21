'''Contain functions of reanding CSI. Pay attention to C-lib path!'''
lib_path = './readcsi.dll' # see .c-source file to find command for compilation


from os import listdir, path
from re import compile
from struct import unpack
from copy import copy

import ctypes
import numpy as np
import pandas as pd


# ---------- LOW-LEVEL ----------
# Preparation to reading .dat files for more speed
lib = ctypes.CDLL(lib_path)
lib.read_csi.restype = None
lib.read_csi.argtypes = (ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(
    ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))

nr, nc, num_tones = 2, 2, 56
csi_re = [[0 for i in range(num_tones)] for k in range(nr * nc)]
csi_im = [[0 for i in range(num_tones)] for k in range(nr * nc)]
for i in range(nr*nc):
    csi_re[i] = (ctypes.c_int * len(csi_re[i]))(*csi_re[i])
    csi_im[i] = (ctypes.c_int * len(csi_im[i]))(*csi_im[i])


def read_csi(csi_buf, nr, nc, num_tones):
    '''Decrypts csi from csi_buf, calling c-function using ctypes'''

    if nr != 2 or nc != 2 or num_tones != 56:
        raise TypeError('Error: nr != 2 or nc != 2 or num_tones != 56')

    csi_buf = (ctypes.c_ubyte * len(csi_buf))(*csi_buf)
    global lib, csi_re, csi_im
    lib.read_csi(csi_buf, csi_re[0], csi_re[1],
                 csi_re[2], csi_re[3], csi_im[0], csi_im[1], csi_im[2], csi_im[3])

    return {'csi_on_path_1': np.array(csi_re[0][:]) + 1j*np.array(csi_im[0][:]), 'csi_on_path_2': np.array(csi_re[1][:]) + 1j * np.array(csi_im[1][:]),
            'csi_on_path_3': np.array(csi_re[2][:]) + 1j*np.array(csi_im[2][:]), 'csi_on_path_4': np.array(csi_re[3][:]) + 1j * np.array(csi_im[3][:])}


def read_log_file(filename, object_type, payload_on, filter_payload=True):
    '''Reads a binary file with CSI.

     object_type - string, object type, for example, bottle or air. payload_on
     - a flag indicating whether payload should be written - payload is usually not needed,
     and its reading time into the array is quite long. filter payload - true: drop packages if payload_len < 1000.
     (Special packages have 1056 payload_len in my case)'''

    with open(filename, 'rb') as f:
        len_file = path.getsize(filename)
        ret, cur, csi_matrix = [], 0, {}

        while cur < (len_file - 4):
            field_len = unpack('>H', f.read(2))[0]
            cur += 2
            if cur + field_len > len_file:
                break

            csi_matrix['timestamp'], csi_matrix['csi_len'], csi_matrix['tx_channel'], csi_matrix['err_info'], csi_matrix['noise_floor'], csi_matrix['rate'], csi_matrix['bandWitdh'], csi_matrix['num_tones'], csi_matrix['nr'], csi_matrix['nc'], csi_matrix['rssi'], csi_matrix['rssi1'], csi_matrix['rssi2'], csi_matrix['rssi3'], csi_matrix['payload_len'] = unpack(
                '>QHHBBBBBBBBBBBH', f.read(25))
            cur += 25

            if csi_matrix['csi_len'] > 0:
                # This condition saves a decent amount of time
                if csi_matrix['csi_len'] == 560:
                    csi_buf = list(unpack('B' * 560, f.read(560)))
                else:
                    csi_buf = [unpack('B', f.read(1))[0]
                               for i in range(csi_matrix['csi_len'])] #TODO почему нельзя написать так: unpack('B'*100, fread(100))

                csi_matrix.update(read_csi(
                    csi_buf, csi_matrix['nr'], csi_matrix['nc'], csi_matrix['num_tones']))
                cur += csi_matrix['csi_len']
            else:
                csi_matrix['csi'] = 0

            if csi_matrix['payload_len'] > 0:
                # It can be done as with csi with csi_len = 560, but payload_len is not always equal to 1040
                if payload_on:
                    csi_matrix['payload'] = [
                        unpack('B', f.read(1))[0] for i in range(csi_matrix['payload_len'])]
                    cur += csi_matrix['payload_len'] # ! оптимизировать
                else:
                    f.read(csi_matrix['payload_len'])
                    cur += csi_matrix['payload_len'] # ! оптимизировать
            else:
                csi_matrix['payload'] = 0

            if (cur + 420) > len_file:
                break
            csi_matrix['object_type'] = object_type
            if (not filter_payload) or csi_matrix['payload_len'] >= 1000: # Filter other recieved Wi-Fi packages (special SCI packages have payload_len ~ 1050)
                ret.append(copy(csi_matrix))
    return ret  # For some reason, the last packet is discarded in the source - here I do not

# ---------- MIDDLE-LEVEL ----------
def set_files_in_groups(file_path, groups):
    '''Split files in file_path into groups'''
    filenames = listdir(file_path)
    new_groups = {}
    for gr in groups:
        newlist = list(filter(compile(gr).match, filenames))
        for i in range(len(newlist)):
            newlist[i] = path.join(file_path, newlist[i])
        new_groups[groups[gr]] = newlist
    return new_groups
    
def get_data(file_groups, payload_on=False):
    dcolumns = ['csi_on_path_' + str(path_num + 1)
                for path_num in range(nr * nc)]
    dcolumns += ['csi_len', 'err_info', 'nc', 'nr', 'num_tones', 'noise_floor',
                 'tx_channel', 'rate', 'rssi1', 'rssi2', 'rssi3', 'timestamp', 'payload_len', 'object_type', 'file_id']
    if payload_on:
        dcolumns += ['payload']

    df_full_csi = pd.DataFrame(columns=dcolumns)
    for group_name in file_groups:
        for filename in file_groups[group_name]:
            data = read_log_file(filename, group_name, payload_on)
            tmp_df = pd.DataFrame(data)
            df_full_csi = pd.concat([df_full_csi, tmp_df])

    return df_full_csi.reset_index(drop=True)


# ---------- HIGH-LEVEL ----------
def get_csi_dfs(big_df):
    '''Returns CSI DataFrames as an array, whose
    length is the number of paths between antennas'''
    csi_columns = [i for i in list(big_df) if 'csi_on_path_' in i]

    csi_dfs = []
    for col in csi_columns:
        new_df = pd.DataFrame(np.vstack(big_df[col]))
        new_df['object_type'] = big_df['object_type']
        csi_dfs.append(new_df)

    return csi_dfs


def make_abs_to_csi_dfs(complex_csi_dfs):
    '''Returns a CSI-DataFrame array in which complex
     numbers are replaced by their modules'''

    abs_csi_dfs = []
    for one_df in complex_csi_dfs:
        abs_csi_dfs.append(one_df.drop(columns=['object_type']).abs().assign(object_type=one_df['object_type'].values))

    return abs_csi_dfs


def get_abs_csi_dfs(filepath, groups):
    '''Wrapper. Reads CSI from files, returns a CSI-DataFrame array,
     which contains the amplitude values of CSI'''

    file_groups = set_files_in_groups(filepath, groups)
    full_df = get_data(file_groups)
    csi_dfs = get_csi_dfs(full_df)
    return make_abs_to_csi_dfs(csi_dfs)


def get_abs_csi_df_big(filepath, groups):
    '''Returns the generic DataFrame in which are written
     in a row to line 56 * 4 = 224 CSI amplitudes and object type (group)'''
    csi_dfs = get_abs_csi_dfs(filepath, groups)
    # ! здесь можно вызвать concat из prep
    type_ds = csi_dfs[0]['object_type']
    for i in range(len(csi_dfs)):
        csi_dfs[i] = csi_dfs[i].drop(['object_type'], axis=1)

    big_df = pd.concat(csi_dfs, axis=1)
    big_df.columns = [i for i in range(0, len(csi_dfs) * csi_dfs[0].shape[1])]

    return big_df.assign(object_type=type_ds)
    # TODO посмотреть, зачем нужна эта функция, она вроде дублирует concat в prep
