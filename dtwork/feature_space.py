'''This file contains funcs which return Series'''

import pandas as pd


def mean(df):
    data = df.drop(columns='object_type')
    data['mean'] = data.mean(axis=1)
    return data.assign(object_type=df['object_type'].values)


def std(df):
    data = df.drop(columns='object_type')
    data['std'] = data.std(axis=1)
    return data.assign(object_type=df['object_type'].values)


def all(df):
    data1 = df.drop(columns='object_type')
    data = data1
    data['std'] = data1.std(axis=1)
    data['mean'] = data1.mean(axis=1)
    data['kurt'] = data1.kurt(axis=1)
    data['skew'] = data1.skew(axis=1)
    data['sem'] = data1.sem(axis=1)
    data['median'] = data1.median(axis=1)
    data['sum'] = data1.sum(axis=1)
    #data['prod'] = data.prod(axis=1)
    data['mad'] = data1.mad(axis=1)

    return data.assign(object_type=df['object_type'].values)


def all_uniq(df, *df_lst, union=True):
    data_in = df.drop(columns='object_type')
    if union:
        data = data_in
    else:
        data = pd.DataFrame()
    data['std_1'] = data_in.std(axis=1)
    data['mean_1'] = data_in.mean(axis=1)
    data['kurt_1'] = data_in.kurt(axis=1)
    data['skew_1'] = data_in.skew(axis=1)
    data['sem_1'] = data_in.sem(axis=1)
    data['median_1'] = data_in.median(axis=1)
    data['sum_1'] = data_in.sum(axis=1)
    data['mad_1'] = data_in.mad(axis=1)

    if len(df_lst) == 0:
        return data.assign(object_type=df['object_type'].values)

    data_lst = [data]
    i = 1
    for df in df_lst:
        i += 1
        data_in = df.drop(columns='object_type')
        if union:
            data = data_in
        else:
            data = pd.DataFrame()
        data['std_' + str(i)] = data_in.std(axis=1)
        data['mean_' + str(i)] = data_in.mean(axis=1)
        data['kurt_' + str(i)] = data_in.kurt(axis=1)
        data['skew_' + str(i)] = data_in.skew(axis=1)
        data['sem_' + str(i)] = data_in.sem(axis=1)
        data['median_' + str(i)] = data_in.median(axis=1)
        data['sum_' + str(i)] = data_in.sum(axis=1)
        data['mad_' + str(i)] = data_in.mad(axis=1)

        data_lst.append(data)
    return data_lst
