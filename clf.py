'''This is the main file of project.'''
# ---------------------------------------- MODULE IMPORTS ----------
# Settings:
make_smooth = True         # Smoothing df. You can set the width of smooth window in code below
make_reduce = False         # Reduce the size of df
make_same = False           # Cutting packets to have the same sizes of all target values
ignore_warnings = True     # For ignore all warnings, use it only if you sure

if ignore_warnings:
    import warnings, os
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from time import time
time_start = time()

import matplotlib.pyplot as plt
from os import path
from dtwork import features
from dtwork import readcsi
from dtwork import prep
from dtwork import plot
from dtwork import ml

import pandas as pd
import numpy as np

np.random.seed(42) # Set seed for repeatability of results
print('Imports complete -->', round(time() - time_start, 2))

# ---------------------------------------- READING ----------
complex_part = 'abs'                                           # 'abs' or 'phase' will reading
groups = {                                                     # Use regex. Only exist groups will be added
    '.*itch.*': 'kitchen',
    'room.*': 'room',
    '.*bathroom.*': 'bathroom',
    'hall.*': 'hall',
    'toilet.*': 'toilet',
    '.*air.*': 'air',
    '.*bottle.*': 'bottle'
}

main_path = path.join('csi', 'homelocation', 'two place')
train_path = path.join(main_path, 'train')
test_path = path.join(main_path, 'test')

df_train = prep.concat_csi(readcsi.get_csi_dfs(train_path, groups, complex_part))
df_test = prep.concat_csi(readcsi.get_csi_dfs(test_path, groups, complex_part))

print('Train packets number:\t', df_train.shape[0], 'Groups:', df_train['object_type'].unique())
print('Test packets number:\t', df_test.shape[0], 'Groups:', df_test['object_type'].unique())
print('Reading complete -->', round(time() - time_start, 2))

# ---------------------------------------- PREPARATION ----------
if make_smooth:
    window = 2  # Smoothing window width
    df_train = prep.concat_csi(prep.smooth(*prep.split_csi(df_train), window=window))
    df_test = prep.concat_csi(prep.smooth(*prep.split_csi(df_test), window=window))

if make_reduce:
    df_train = prep.decimate_one(df_train, 5, 7, 9, 11, 13)
    print('New df_train size:', df_train.shape[0])

if make_same:
    df_train = prep.make_same(df_train)
    df_test = prep.make_same(df_test)

# ---------------------------------------- CLASSIFICATION ----------
clf_res = pd.concat([
    #ml.fit_ffnn(df_train, df_test),
    ml.fit_cnn(df_train, df_test),
    ml.fit_sklearn(df_train, df_test),
    ])


# ---------------------------------------- RESULTS COMPARISON ----------
sorted_res = clf_res.sort_values(by='accuracy', ascending=False, ignore_index=True)
print('\nClassification results:')
print(sorted_res)
sorted_res.to_csv('results\\ml_results.csv', index=False)
print('Finish -->', round(time() - time_start, 2))