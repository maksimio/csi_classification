'''This is the main file of project.'''

# ---------------------------------------- MODULE IMPORTS ----------
from time import time
time_start = time()

from sklearn.feature_selection import SelectKBest, chi2
from dtwork import ml
from dtwork import plot
from dtwork import prep
from dtwork import readcsi
from dtwork import feature_space
from os import path
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from dtwork.feature_selector import FeatureSelector

# Settings:
only_plot = False          #
select_features = False    #
use_keras = True          #
learn_all_pathes = False   #

make_smooth = False        # Smoothing df. You can set the width of smooth window in code below
make_reduce = False        # Reduce the size of df:
make_same   = True        # Cutting packets to have the same sizes of all target values:

print('Imports complete -->', round(time() - time_start, 2))

# ---------------------------------------- READING ----------
dirs_to_groups = ['csi', 'homelocation', 'two place']
#dirs_to_groups = ['csi', 'other', '2_objects', '250 cm']
complex_part = 'abs' # 'abs' or 'phase' will reading
groups = { # use regex
    '.*itch.*': 'kitchen',
    'room.*': 'room',
    '.*bathroom.*':'bathroom',
    'hall.*':'hall',
    'toilet.*':'toilet',
    '.*air.*': 'air',
    '.*bottle.*': 'bottle'
}

main_path = path.join(*dirs_to_groups)
train_path = path.join(main_path, 'train')
test_path = path.join(main_path, 'test')

df_train = prep.concat_csi(readcsi.get_csi_dfs(train_path, groups, complex_part))
df_test = prep.concat_csi(readcsi.get_csi_dfs(test_path, groups, complex_part))

print('Train packets number:\t', df_train.shape[0])
print('Test packets number:\t', df_test.shape[0])
print('Reading complete -->', round(time() - time_start, 2))
print('Found groups df_train:', df_train['object_type'].unique())
print('Found groups df_test:', df_test['object_type'].unique())

# ---------------------------------------- PREPARATION ----------
if make_smooth:
  window = 2  # smoothing window width
  df_train = prep.concat_csi(prep.smooth(*prep.split_csi(df_train), window=window))
  df_test = prep.concat_csi(prep.smooth(*prep.split_csi(df_test), window=window))

if make_reduce:
  df_train = prep.decimate_one(df_train, 5, 7, 9, 11, 13)
  print('New df_train size:', df_train.shape[0])

if make_same:
  df_train = prep.make_same(df_train)
  df_test = prep.make_same(df_test)

if only_plot:
  #df_train = prep.concat_csi(prep.down(*prep.split_csi(df_train)))
  #df_train_temp = df_train.drop(columns='object_type') < 3.14
  #print(df_train_temp)
  #df_train = df_train_temp.assign(object_type=df_train['object_type'].values)
  plot.plot_examples(df_train)
  exit()


#df_train = pd.concat([df_train, *feature_space.all_uniq(*prep.split_csi(df_train), union=False)], axis=1)
#df_test = pd.concat([df_test, *feature_space.all_uniq(*prep.split_csi(df_test), union=False)], axis=1)

#####df_train = df_train.drop(['object_type'], axis=1).diff(axis=1).fillna(0).assign(object_type=df_train['object_type'].values) # GOOD
#####df_test = df_test.drop(['object_type'], axis=1).diff(axis=1).fillna(0).assign(object_type=df_test['object_type'].values)    # GOOD

#plot.plot_examples(df_train)
#print(df_train)
#sns.scatterplot(x='skew_1', y='std_1', hue='object_type', data=df_train, alpha=0.44)
#plt.grid()
#plt.show()
# df_train = prep.split_csi(df_train)
# df_train = prep.concat_csi([df_train[0], df_train[2], df_train[1], df_train[3]])
# df_test = prep.split_csi(df_test)
# df_test = prep.concat_csi([df_test[0], df_test[2], df_test[1], df_test[3]])

ml.cnn(df_train, df_test)
exit()
#!---------------------------------------------- Feature selection
data = df_train.drop(['object_type'], axis=1)[[i * 10 for i in range(220 // 10)]]
labels = df_train['object_type']
fs = FeatureSelector(data=data, labels=labels)
fs.identify_collinear(correlation_threshold=0.95)
fs.plot_collinear(plot_all=True)
fs.identify_zero_importance('classification', 'auc')
fs.plot_feature_importances(threshold = 0.95, plot_n = 15)
plt.show()
fs.identify_low_importance(cumulative_importance = 0.90)
print(fs.feature_importances)
print('ssssssssssssssssssssssssssssssss')
print(fs.record_low_importance)

# print(fs.record_collinear)
exit()
#!----------------------------------------------



# Prepare
x_train = df_train.drop('object_type', axis=1)
y_train = df_train['object_type']
x_test = df_test.drop('object_type', axis=1)
y_test = df_test['object_type']

# ---------------------------------------- FEATURE SELECTION ----------
if select_features:
  kBest = SelectKBest(score_func=chi2, k=10)
  # absolute values depend of df size
  scores_train = kBest.fit(x_train, y_train).scores_
  scores_test = kBest.fit(x_test, y_test).scores_

  df = pd.DataFrame(pd.Series(scores_train, name='train'), pd.Series(x_train.columns, name='subcarriers'))
  df['train_%'] = (df['train'] / df['train'].max() * 100).astype(int)
  df['test'] = pd.Series(scores_test, name='test')
  df['test_%'] = (df['test'] / df['test'].max() * 100).astype(int)
  df['sig_way'] = [i // 56 + 1 for i in range(224)]
  df['subc_num'] = [i % 56 + 1 for i in range(224)]
  df['test'] = df['test'].astype(int)
  df['train'] = df['train'].astype(int)

  df.to_csv('results\\correlation.csv') # Output

# ---------------------------------------- CLASSIFICATION ----------
clf_res = ml.ml(x_train, y_train, x_test, y_test, df_train.copy(), df_test.copy(), time_start=time_start, use_keras=use_keras)

if learn_all_pathes:
  print('\tML FOR ALL PATHES -->', round(time() - time_start, 2))
  # After see file statictic_correlation.csv we would like
  # to try use ML for every path of signal (max 4)
  dfs_train = prep.split_csi(df_train)
  dfs_test = prep.split_csi(df_test)

  i = 1
  for train, test in zip(dfs_train, dfs_test):
    x_train_1 = train.drop('object_type', axis=1)
    y_train_1 = train['object_type']
    x_test_1 = test.drop('object_type', axis=1)
    y_test_1 = test['object_type']

    clf_res_1 = ml.ml(x_train_1, y_train_1, x_test_1, y_test_1, train.copy(), test.copy(), time_start=time_start, use_keras=use_keras)
    clf_res['acc_'+str(i)] = clf_res_1['accuracy']
    clf_res['time_'+str(i)] = clf_res_1['time']

    i += 1
    print('ML FOR', i, 'PATH -->', round(time() - time_start, 2))

  clf_res['aver_time_1234'] = ((clf_res['time_1'] + clf_res['time_2'] + clf_res['time_3'] + clf_res['time_4']) / 4).round(2)
  clf_res = clf_res.drop(['time_' + str(i + 1) for i in range(4)], axis=1)

# ---------------------------------------- RESULTS COMPARISON ----------
sorted_res = clf_res.sort_values(by='accuracy', ascending=False, ignore_index=True)
print('\nClassification results:')
print(sorted_res)
sorted_res.to_csv('results\\ml_results.csv', index=False)
print('Finish -->', round(time() - time_start, 2))
