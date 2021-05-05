'''This file contains functions, witch work with features of dfs.'''

import pandas as pd


def statistics(df):
    '''Return various statistic DataFrame with
    target value column "object_type"'''

    data = pd.DataFrame()
    df_features = df.drop(columns='object_type')

    data['std'] = df_features.std(axis=1)
    data['mean'] = df_features.mean(axis=1)
    data['kurt'] = df_features.kurt(axis=1)
    data['skew'] = df_features.skew(axis=1)
    data['sem'] = df_features.sem(axis=1)
    data['median'] = df_features.median(axis=1)
    data['mad'] = df_features.mad(axis=1)

    return data


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