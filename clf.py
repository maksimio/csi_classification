# This is the main module
plot_and_exit = False
select_features = True
use_keras = True
learn_all_pathes = False
# ---------- MODULE IMPORTS ----------
from time import time
time_start = time()
import pandas as pd
from os import path
from dtwork import readcsi
from dtwork import prep
from dtwork import plot
from dtwork import ml

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


print('Imports complete -->', round(time() - time_start, 2))

# ---------- DATA READING ----------
# -=-=-=- Enter groups here using regex (expression : group_name):
groups = {
    ".*bottle.*": "bottle",
    ".*air.*": "air"
}
# -=-=-=- Enter your training and test data paths here:
main_path = path.join("csi", "use_in_paper","2_objects")
#main_path = path.join("csi", "other","2_objects","250 cm")

train_path = path.join(main_path,"train")
test_path = path.join(main_path, "test")

df_train = readcsi.get_abs_csi_df_big(train_path,groups)
df_test = readcsi.get_abs_csi_df_big(test_path,groups)
# Use pandas to work with data (example: print(df_train.head(5)))

print('Train packets number:\t', df_train.shape[0])
print('Test packets number:\t', df_test.shape[0])
print('Preparation complete -->', round(time() - time_start, 2))

# ---------- PROCESSING AND VISUALISATION ----------
# There are examples of data processing and visualization,
# including smoothing, reducing the number of packets and graphical representation.
# We use our modules prep and plot here

if plot_and_exit:
    small_df_train = prep.cut_csi(df_train, 100) # To make the schedule faster
    # Simple showing:
    if True:
        plot.csi_plot_types(small_df_train)

    # Showing with smoothing and lowering:
    if True:
        df_lst = prep.split_csi(small_df_train)
        smoothed_df_lst = prep.smooth(*df_lst)
        lowered_df_lst = prep.down(*smoothed_df_lst)
        new_small_df = prep.concat_csi(lowered_df_lst)
        
        plot.csi_plot_types(new_small_df)

    # Wrong showing (smoothing full df):
    if True:
        moothed_df_lst = prep.smooth_savgol(small_df_train)
        plot.csi_plot_types(moothed_df_lst)

    # Showing only one path of antennas:
    if True:
        df_lst = prep.split_csi(small_df_train)
        plot.csi_plot_types(df_lst[3])      

    # Showing smoothed one path and all paths using simple smoothing:
    if True:
        df_lst = prep.split_csi(small_df_train)
        smoothed_df_lst = prep.smooth(*df_lst, window=6)
        plot.csi_plot_types(smoothed_df_lst[0])
        plot.csi_plot_types(prep.concat_csi(smoothed_df_lst))

    exit()
    
# Reduce the size of df:
if False:
    df_train = prep.decimate_one(df_train,5,7,9,11,13)
    print('New df_train size:',df_train.shape[0])

# ---------- DATA PREPARATION ----------
# Cutting packets to have the same sizes of all target values:
df_train = prep.make_same(df_train)
df_test = prep.make_same(df_test)

# Prepare 
x_train = df_train.drop('object_type',axis=1)
y_train = df_train['object_type']
x_test = df_test.drop('object_type',axis=1)
y_test = df_test['object_type']

# ---------- FEATURE SELECTION ----------
if select_features:
    kBest = SelectKBest(score_func=chi2, k=10)
    scores_train = kBest.fit(x_train, y_train).scores_ # values depend of df size
    scores_test = kBest.fit(x_test, y_test).scores_

    df = pd.DataFrame(pd.Series(scores_train,name='train'),pd.Series(x_train.columns,name='subcarriers'))
    df['train_%'] = (df['train']/df['train'].max()*100).astype(int)
    df['test'] = pd.Series(scores_test,name='test')
    df['test_%'] = (df['test']/df['test'].max()*100).astype(int)
    df['sig_way'] = [i//56+1 for i in range(224)]
    df['subc_num'] = [i % 56 + 1 for i in range(224)]
    df['test'] = df['test'].astype(int)
    df['train'] = df['train'].astype(int)

    df.to_csv('results\\statistic_correlation1.csv') # In this file you can see correlation
    # for train and test datasets for all 4 (in our case) ways of signal between antennas

# ---------- CLASSIFICATION ----------
clf_res = ml.ml(x_train,y_train,x_test,y_test,df_train.copy(),df_test.copy(),time_start=time_start,use_keras=True)

if learn_all_pathes:
    print('ML FOR ALL PATHES -->', round(time() - time_start, 2))
    # After see file statictic_correlation.csv we would like
    # to try use ML for every path of signal (max 4)
    dfs_train = prep.split_csi(df_train)
    dfs_test = prep.split_csi(df_train)

    i = 1
    for train, test in zip(dfs_train, dfs_test):
        x_train_1 = train.drop('object_type',axis=1)
        y_train_1 = train['object_type']
        x_test_1 = test.drop('object_type',axis=1)
        y_test_1 = test['object_type']

        clf_res_1 = ml.ml(x_train_1,y_train_1,x_test_1,y_test_1,train.copy(),test.copy(),time_start=time_start,use_keras=use_keras)
        clf_res['acc_'+str(i)] = clf_res_1['accuracy']
        clf_res['time_'+str(i)] = clf_res_1['time']

        i+=1
        print('ML FOR', i,'PATH -->', round(time() - time_start, 2))
    
    clf_res['aver_time_1234'] = ((clf_res['time_1']+clf_res['time_2']+clf_res['time_3']+clf_res['time_4'])/4).round(2)
    clf_res = clf_res.drop(['time_'+str(i+1) for i in range(4)],axis=1)

# ---------- RESULTS COMPARISON ----------
sorted_res = clf_res.sort_values('accuracy',ascending=False).reset_index()
print('Classification results:')
print(sorted_res)
sorted_res.to_csv('results\\ml_results.csv',index=False)
print('Finish -->', round(time() - time_start, 2))
