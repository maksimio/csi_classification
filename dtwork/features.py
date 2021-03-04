'''This file contains functions, witch work with features of dfs.'''

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


# ---------------------------------------- FEATURE SELECTION ----------
# if select_features:
#     kBest = SelectKBest(score_func=chi2, k=10)
#     # absolute values depend of df size
#     scores_train = kBest.fit(x_train, y_train).scores_
#     scores_test = kBest.fit(x_test, y_test).scores_

#     df = pd.DataFrame(pd.Series(scores_train, name='train'),
#                       pd.Series(x_train.columns, name='subcarriers'))
#     df['train_%'] = (df['train'] / df['train'].max() * 100).astype(int)
#     df['test'] = pd.Series(scores_test, name='test')
#     df['test_%'] = (df['test'] / df['test'].max() * 100).astype(int)
#     df['sig_way'] = [i // 56 + 1 for i in range(224)]
#     df['subc_num'] = [i % 56 + 1 for i in range(224)]
#     df['test'] = df['test'].astype(int)
#     df['train'] = df['train'].astype(int)

#     df.to_csv('results\\correlation.csv')  # Output



# #!---------------------------------------------- Feature selection
##from sklearn.feature_selection import SelectKBest, chi2 #! delete
# # [[i * 10 for i in range(220 // 10)]]
# data = df_train.drop(['object_type'], axis=1)
# labels = df_train['object_type']
# fs = FeatureSelector(data=data, labels=labels)
# fs.identify_collinear(correlation_threshold=0.95)
# fs.plot_collinear(plot_all=True)
# fs.identify_zero_importance('classification', 'auc')
# fs.plot_feature_importances(threshold=0.95, plot_n=15)
# plt.show()
# fs.identify_low_importance(cumulative_importance=0.90)
# print(fs.feature_importances)
# print('ssssssssssssssssssssssssssssssss')
# print(fs.record_low_importance)

# print(fs.record_collinear)
# exit()
# #!----------------------------------------------





#---------------------------------------DIFF
# df_train = df_train.drop(['object_type'], axis=1).diff(axis=1).fillna(0).assign(object_type=df_train['object_type'].values) # GOOD
# df_test = df_test.drop(['object_type'], axis=1).diff(axis=1).fillna(0).assign(object_type=df_test['object_type'].values)    # GOOD











#df_train = pd.concat([df_train, *feature_space.all_uniq(*prep.split_csi(df_train), union=False)], axis=1)
#df_test = pd.concat([df_test, *feature_space.all_uniq(*prep.split_csi(df_test), union=False)], axis=1)

# plot.plot_examples(df_train)
# print(df_train)
#sns.scatterplot(x='skew_1', y='std_1', hue='object_type', data=df_train, alpha=0.44)
# plt.grid()
# plt.show()
# df_train = prep.split_csi(df_train)
# df_train = prep.concat_csi([df_train[0], df_train[2], df_train[1], df_train[3]])
# df_test = prep.split_csi(df_test)
# df_test = prep.concat_csi([df_test[0], df_test[2], df_test[1], df_test[3]])