# This is the main module
from sklearn.feature_selection import SelectKBest, chi2
from dtwork import ml
from dtwork import plot
from dtwork import prep
from dtwork import readcsi
from os import path
import pandas as pd
from time import time

plot_and_exit = True
select_features = False
use_keras = False
learn_all_pathes = True
make_smooth = False

# ---------- MODULE IMPORTS ----------
time_start = time()


print('Imports complete -->', round(time() - time_start, 2))

# ---------- DATA READING ----------
# -=-=-=- Enter groups here using regex (expression : group_name):
groups = {
    # '.*itch.*': 'kitchen',
    # 'room.*': 'room',
    # '.*bathroom.*':'bathroom',
    # 'hall.*':'hall',
    # 'toilet.*':'toilet',
    '.*air.*': 'air',
    '.*bottle.*': 'bottle'
}
# -=-=-=- Enter your training and test data paths here:
#main_path = path.join('csi', 'use_in_paper', '2_objects')
main_path = path.join('csi', 'other', '2_objects', '250 cm')
#main_path = path.join('csi', 'homelocation', 'two place')

train_path = path.join(main_path, 'train')
test_path = path.join(main_path, 'test')

df_train = readcsi.get_abs_csi_df_big(train_path, groups)
df_test = readcsi.get_abs_csi_df_big(test_path, groups)
# Use pandas to work with data (example: print(df_train.head(5)))

print('Train packets number:\t', df_train.shape[0])
print('Test packets number:\t', df_test.shape[0])
print('Preparation complete -->', round(time() - time_start, 2))

# ---------- PROCESSING AND VISUALISATION ----------
# There are examples of data processing and visualization,
# including smoothing, reducing the number of packets and graphical representation.
# We use our modules prep and plot here

if plot_and_exit:  # TODO check plot code
    small_df_train = prep.cut_csi(df_train, 30)  # To make the schedule faster

    if False:  # Simple showing:
        plot.csi_plot_types(small_df_train)

    if True:  # Showing with smoothing and lowering:
        df_lst = prep.split_csi(small_df_train)
        smoothed_df_lst = prep.smooth(*df_lst)
        lowered_df_lst = prep.down2(*smoothed_df_lst, level=150)
        new_small_df = prep.concat_csi(lowered_df_lst)
        plot.csi_plot_types(new_small_df)

    if True:  # Wrong showing (smoothing full df):
        moothed_df_lst = prep.smooth_savgol(small_df_train)
        plot.csi_plot_types(moothed_df_lst)

    if True:  # Showing only one path of antennas:
        df_lst = prep.split_csi(small_df_train)
        plot.csi_plot_types(df_lst[3])

    if True:  # Showing smoothed one path and all paths using simple smoothing:
        df_lst = prep.split_csi(small_df_train)
        smoothed_df_lst = prep.smooth(*df_lst, window=2)
        plot.csi_plot_types(smoothed_df_lst[0])
        plot.csi_plot_types(prep.concat_csi(smoothed_df_lst))

    exit()

if make_smooth:  # Smoothing dfs:
    window = 2  # smoothing window width
    df_train = prep.concat_csi(prep.smooth(
        *prep.split_csi(df_train), window=window))
    df_test = prep.concat_csi(prep.smooth(
        *prep.split_csi(df_test), window=window))

if False:  # Reduce the size of df:
    df_train = prep.decimate_one(df_train, 5, 7, 9, 11, 13)
    print('New df_train size:', df_train.shape[0])

# ---------- DATA PREPARATION ----------
# Cutting packets to have the same sizes of all target values:
df_train = prep.make_same(df_train)
df_test = prep.make_same(df_test)

# Prepare
x_train = df_train.drop('object_type', axis=1)
y_train = df_train['object_type']
x_test = df_test.drop('object_type', axis=1)
y_test = df_test['object_type']

# ---------- FEATURE SELECTION ----------
if select_features:
    kBest = SelectKBest(score_func=chi2, k=10)
    # absolute values depend of df size
    scores_train = kBest.fit(x_train, y_train).scores_
    scores_test = kBest.fit(x_test, y_test).scores_

    df = pd.DataFrame(pd.Series(scores_train, name='train'),
                      pd.Series(x_train.columns, name='subcarriers'))
    df['train_%'] = (df['train'] / df['train'].max() * 100).astype(int)
    df['test'] = pd.Series(scores_test, name='test')
    df['test_%'] = (df['test'] / df['test'].max() * 100).astype(int)
    df['sig_way'] = [i // 56+1 for i in range(224)]
    df['subc_num'] = [i % 56 + 1 for i in range(224)]
    df['test'] = df['test'].astype(int)
    df['train'] = df['train'].astype(int)

    # In this file you can see correlation
    df.to_csv('results\\statistic_correlation1.csv')
    # for train and test datasets for all 4 (in our case) ways of signal between antennas

# ---------- CLASSIFICATION ----------
clf_res = ml.ml(x_train, y_train, x_test, y_test, df_train.copy(),
                df_test.copy(), time_start=time_start, use_keras=use_keras)

if learn_all_pathes:
    print('\tML FOR ALL PATHES -->', round(time() - time_start, 2))
    # After see file statictic_correlation.csv we would like
    # to try use ML for every path of signal (max 4)
    dfs_train = prep.split_csi(df_train)
    dfs_test = prep.split_csi(df_test)  # !!!

    i = 1
    for train, test in zip(dfs_train, dfs_test):
        x_train_1 = train.drop('object_type', axis=1)
        y_train_1 = train['object_type']
        x_test_1 = test.drop('object_type', axis=1)
        y_test_1 = test['object_type']

        clf_res_1 = ml.ml(x_train_1, y_train_1, x_test_1, y_test_1, train.copy(
        ), test.copy(), time_start=time_start, use_keras=use_keras)
        clf_res['acc_'+str(i)] = clf_res_1['accuracy']
        clf_res['time_'+str(i)] = clf_res_1['time']

        i += 1
        print('ML FOR', i, 'PATH -->', round(time() - time_start, 2))

    clf_res['aver_time_1234'] = (
        (clf_res['time_1'] + clf_res['time_2'] + clf_res['time_3'] + clf_res['time_4']) / 4).round(2)
    clf_res = clf_res.drop(['time_' + str(i + 1) for i in range(4)], axis=1)

# ---------- RESULTS COMPARISON ----------
sorted_res = clf_res.sort_values(by='accuracy', ascending=False).reset_index()
print('Classification results:')
print(sorted_res)
sorted_res.to_csv('results\\ml_results.csv', index=False)
print('Finish -->', round(time() - time_start, 2))
# TODO add average percent and reset index when sort
