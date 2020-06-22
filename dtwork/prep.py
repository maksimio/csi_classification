'''Модуль, в котором содержаться функции компановки df из CSI,
а также функции предобработки данных. Имеются функции
децимации, сглаживания.'''

import pandas as pd
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------КОМПАНОВКА DF


def split_csi(big_df, num_tones=56):
    '''Возвращает массив CSI df, по num_tones амплитуд
    в каждом df. Противоположна concat_csi(...). 
    Принимает Big_df, в котором записны в строку
    num_tones*4=224 поднесущих CSI.'''
    df_lst = []
    for k in range(4):
        one_df = big_df[[i+k*num_tones for i in range(0, num_tones)]]
        one_df.columns = [i for i in range(0, num_tones)]
        one_df = one_df.assign(object_type=big_df['object_type'].values)
        df_lst.append(one_df)
    return df_lst


def concat_csi(df_lst):
    '''Возвращает общий DataFrame, в котором записаны
    подряд в строку 56*4=224 поднесущих CSI'''
    type_ds = df_lst[0]['object_type']
    for i in range(len(df_lst)):
        df_lst[i] = df_lst[i].drop(['object_type'], axis=1)
    big_df = pd.concat(df_lst, axis=1)
    big_df.columns = [i for i in range(0, len(df_lst)*df_lst[0].shape[1])]
    return big_df.assign(object_type=type_ds)


# -----------------------------------------------------------------------ПРЕДОБРАБОТКА DF


def down(df, *df_lst):
    '''Опускает амплитуды csi, вычитая из каждого пакета
    минимальное значение. Рекомендуется использовать индивидуально
    к пакетам с каждого пути, а не к склеенному df.'''
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


def cut_csi(df, number, shuffle: bool=True):
    '''Возвращает dataframe, в котором осталось
    number пакетов для каждого объекта. Shuffle - 
    выбрать пакеты случайным образом. Только для
    склеенного df при shuffle=True!'''
    df_lst = []
    object_types = df['object_type'].unique()
    for obj_type in object_types:
        obj_df = df[df['object_type'] == obj_type] 
        if shuffle:
            obj_df = obj_df.sample(frac=1).reset_index(drop=True)   
        df_lst.append(obj_df.head(number))
    return pd.concat(df_lst, axis=0).reset_index(drop=True)


def decimate_one(df, k, *k_lst):
    '''Удаляет каждую k-тую строку. Можно удалить любые
    кратные строки из df, передав несколько
    аргументов. Кратные строки могут пересекаться, их повторы
    будут удалены перед drop. Таким образом, при передаче
    (df, 2, 2) в функцию, от исходного df останется 1/2.'''
    drop_index_lst = [i for i in range(0, df.shape[0], k)]
    for k in k_lst:
        drop_index_lst += [i for i in range(0, df.shape[0], k)]
    drop_index_lst = list(set(drop_index_lst))
    return df.drop(drop_index_lst).reset_index(drop=True)


def decimate_every(df, k, *k_lst):
    '''Удаляет каждую k-ю строку. Каждый раз после удаления
    строки из df в нем сбрасываются индексы. Таким образом, при
    передаче (df, 2, 2) в функцию, от исходного df останется 1/4.'''
    drop_index_lst = [i for i in range(0, df.shape[0], k)]
    df = df.drop(drop_index_lst).reset_index(drop=True)
    for k in k_lst:
        drop_index_lst = [i for i in range(0, df.shape[0], k)]
        df = df.drop(drop_index_lst).reset_index(drop=True)
    return df


def smooth_csi(df, *df_lst, win_width=9, polyorder=3):
    '''Cглаживает csi. Не рекомендуется
    применять к склеенному df. Применяется фильтр
    Савицкого-Голея.'''
    smoothed = savgol_filter(
        df.drop(columns='object_type'), win_width, polyorder)
    if len(df_lst) == 0:
        return pd.DataFrame(smoothed).assign(object_type=df['object_type'].values)
    else:
        smoothed_lst = [pd.DataFrame(smoothed).assign(
            object_type=df['object_type'].values)]
        for df in df_lst:
            smoothed = savgol_filter(
                df.drop(columns='object_type'), win_width, polyorder)
            smoothed_lst.append(pd.DataFrame(smoothed).assign(
                object_type=df['object_type'].values))
        return smoothed_lst
