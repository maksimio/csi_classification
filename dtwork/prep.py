'''A module that contains df linking functions from CSI,
as well as data preprocessing functions. There are functions
decimation, smoothing.'''

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

# ---------- GROUPING ----------
def split_csi(big_df, num_tones=56):
    '''Returns an array of CSI df, by num_tones amplitudes
    in every df. The opposite of concat_csi (...).
    Accepts Big_df in which are written to a string
    num_tones * 4 = 224 CSI subcarriers'''

    df_lst = []
    for k in range(4):
        one_df = big_df[[i+k*num_tones for i in range(0, num_tones)]]
        one_df.columns = [i for i in range(0, num_tones)]
        one_df = one_df.assign(object_type=big_df['object_type'].values)
        df_lst.append(one_df)
    return df_lst


def concat_csi(df_lst):
    '''Returns the generic DataFrame in which are written
    row 56 * 4 = 224 CSI subcarriers'''

    type_ds = df_lst[0]['object_type']
    for i in range(len(df_lst)):
        df_lst[i] = df_lst[i].drop(['object_type'], axis=1)
    big_df = pd.concat(df_lst, axis=1)
    big_df.columns = [i for i in range(0, len(df_lst)*df_lst[0].shape[1])]
    return big_df.assign(object_type=type_ds)


# ---------- CHANGE ----------
def down(df, *df_lst):
    '''Lowers csi amplitudes by subtracting from each packet
    minimum value. It is recommended to use individually.
    to packets from each path, and not to the glued df'''

    object_type = df['object_type']
    min_col = df.drop(['object_type'], axis=1).min(axis=1)
    df_down = df.drop(['object_type'], axis=1).sub(min_col, axis=0)
    df_down['object_type'] = object_type
    if len(df_lst) == 0:
        return df_down
    else:
        df_down_lst = [df_down]
        for df in df_lst:
            min_col = df.drop(['object_type'], axis=1).min(axis=1)
            df_down = df.drop(['object_type'], axis=1).sub(min_col, axis=0)
            df_down['object_type'] = object_type
            df_down_lst.append(df_down)
        return df_down_lst


def smooth_savgol(df, *df_lst, win_width=9, polyorder=3):
    '''Smoothes csi. Not recommended apply to glued df. 
    Filter applied Savitsky-Golay'''

    smoothed = savgol_filter(df.drop(columns='object_type'), win_width, polyorder)
    if len(df_lst) == 0:
        return pd.DataFrame(smoothed).assign(object_type=df['object_type'].values)
    else:
        smoothed_lst = [pd.DataFrame(smoothed).assign(
            object_type=df['object_type'].values)]
        for df in df_lst:
            smoothed = savgol_filter(df.drop(columns='object_type'), win_width, polyorder)
            smoothed_lst.append(pd.DataFrame(smoothed).assign(object_type=df['object_type'].values))
        return smoothed_lst


def smooth(df, *df_lst, window=5, win_type=None):
    '''Smoothes csi. See about possible win_type's here:
    https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows
    For example, use "hamming" win_type.'''

    smoothed = df.drop(columns='object_type').T.rolling(window, min_periods=1, center=True, win_type=win_type).mean().T
    if len(df_lst) == 0:
        return smoothed.assign(object_type=df['object_type'].values)
    else:
        smoothed_lst = [smoothed.assign(object_type=df['object_type'].values)]
        for df in df_lst:
            smoothed = df.drop(columns='object_type').T.rolling(window, min_periods=1, center=True, win_type=win_type).mean().T
            smoothed_lst.append(smoothed.assign(object_type=df['object_type'].values))
        return smoothed_lst


def normalize_phase_old(df, *df_lst):
    '''Deprecated - too slow. Use "normalize_phase".
    Remove jumps when phases crossing [-pi; pi]'''

    df_new = df.copy()
    for i in range(df_new.shape[0]):
        shift = 0
        for j in range(df_new.shape[1] - 1):
            df.loc[i, j] = df_new.loc[i, j] + shift
            if j == 55: #TODO: !
                break
            if df_new.loc[i, j] - df_new.loc[i, j + 1] > 3:
                shift += np.pi * 2
            elif df_new.loc[i, j + 1] - df_new.loc[i, j] > 3:
                shift -= np.pi * 2
    
    if len(df_lst) == 0:
        return df
    else:
        normalize_lst = [df]
        for df in df_lst:
            df_new = df.copy()
            for i in range(df_new.shape[0]):
                shift = 0
                for j in range(df_new.shape[1] - 1):
                    df.loc[i, j] = df_new.loc[i, j] + shift
                    if j == 55:
                        break
                    if df_new.loc[i, j] - df_new.loc[i, j + 1] > 3:
                        shift += np.pi * 2
                    elif df_new.loc[i, j + 1] - df_new.loc[i, j] > 3:
                        shift -= np.pi * 2
            normalize_lst.append(df)
        return normalize_lst


def normalize_phase(df, *df_lst):
    '''Remove jumps when phases crossing [-pi; pi] in radians'''
    
    df_target = df['object_type']
    df.drop('object_type', axis=1, inplace=True)

    df_diff = df.diff(axis=1).fillna(0)
    df_diff[(df_diff < np.pi * 2 - 0.5) & (df_diff > -np.pi * 2 + 0.5)] = 0

    for column in df:
        df.iloc[:, column:][df_diff[column] > 0] -= np.pi * 2
        df.iloc[:, column:][df_diff[column] < 0] += np.pi * 2
    
    if len(df_lst) == 0:
        return df.assign(object_type=df_target.values)

    normalize_lst = [df.assign(object_type=df_target.values)]
    for df in df_lst:
        df.drop('object_type', axis=1, inplace=True)
        df_diff = df.diff(axis=1).fillna(0)
        df_diff[(df_diff < np.pi * 2 - 1) & (df_diff > -np.pi * 2 + 1)] = 0
        
        for column in df:
            df.iloc[:, column:][df_diff[column] > 0] -= np.pi * 2
            df.iloc[:, column:][df_diff[column] < 0] += np.pi * 2
        normalize_lst.append(df.assign(object_type=df_target.values))
    
    return normalize_lst


def difference(df, *df_lst):
    '''Convert df values to difference between them'''

    diff = df.drop(['object_type'], axis=1).diff(axis=1).fillna(0)
    if len(df_lst) == 0:
        return diff.assign(object_type=df['object_type'].values)
    else:
        diff_lst = [diff.assign(object_type=df['object_type'].values)]
        for df in df_lst:
            diff = df.drop(['object_type'], axis=1).diff(axis=1).fillna(0)
            diff_lst.append(diff.assign(object_type=df['object_type'].values))
        return diff_lst


# ---------- SCREENING ----------
def cut_csi(df, number, shuffle: bool = True):
    '''Returns the dataframe in which is left
    number of packets for each object. Shuffle -
    Choose packages randomly. Only for
    glued df with shuffle = True!'''

    df_lst = []
    object_types = df['object_type'].unique()
    for obj_type in object_types:
        obj_df = df[df['object_type'] == obj_type]
        if shuffle:
            obj_df = obj_df.sample(frac=1).reset_index(drop=True)
        df_lst.append(obj_df.head(number))
    return pd.concat(df_lst, axis=0).reset_index(drop=True)


def decimate_one(df, k, *k_lst):
    '''Deletes every kth row. You can delete any
    multiple lines from df passing multiple
    arguments. Multiple lines may intersect, their repetitions
    will be deleted before drop. Thus, when transmitting
    (df, 2, 2) to the function, 1/2 will remain from the original df.'''

    drop_index_lst = [i for i in range(0, df.shape[0], k)]
    for k in k_lst:
        drop_index_lst += [i for i in range(0, df.shape[0], k)]
    drop_index_lst = list(set(drop_index_lst))
    return df.drop(drop_index_lst).reset_index(drop=True)


def decimate_every(df, k, *k_lst):
    '''Deletes every kth row. Every time after removal
    rows from df in it are reset indices. Thus, when
    passing (df, 2, 2) to the function, 1/4 will remain from the original df.'''

    drop_index_lst = [i for i in range(0, df.shape[0], k)]
    df = df.drop(drop_index_lst).reset_index(drop=True)

    for k in k_lst:
        drop_index_lst = [i for i in range(0, df.shape[0], k)]
        df = df.drop(drop_index_lst).reset_index(drop=True)
    return df


def make_same(df):
    '''Cutting packets to have the same sizes
    of all target values and mixing rows of df.'''

    o_types = pd.unique(df['object_type']).tolist()
    min_len = 100000000000
    df_lst = []

    for o in o_types:
        min_len = min(df[df['object_type'] == o].shape[0], min_len)
    for o in o_types:
        df_lst.append(df[df['object_type'] == o].head(min_len))

    df = pd.concat(df_lst)
    return df.sample(frac=1).reset_index(drop=True)  # Mixing df
