'''This is the main file of project.'''
# ---------------------------------------- MODULE IMPORTS ----------
# Settings:
settings = {
    'make_smooth': True,         # Smoothing df. You can set the width of smooth window in code below
    'make_reduce': False,         # Reduce the size of df
    'make_same': True,           # Cutting packets to have the same sizes of all target values
    'normalize': True,           # Only for phase! Remove phase jumps
    'diff_order': 0,             # Order of derivative (usual difference). 0 means without that option
    'ignore_warnings': True     # For ignore all warnings, use it only if you sure. It speed up the code 
}

if settings['ignore_warnings']:
    import warnings, os
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from time import time
time_start = time()

from os import path
from dtwork import features
from dtwork import readcsi
from dtwork import prep
from dtwork import plot
from dtwork import ml

import pandas as pd

if settings['ignore_warnings']:
    pd.options.mode.chained_assignment = None


print('Imports complete -->', round(time() - time_start, 2))

print('Settings:')
for key in settings:
    print('--- ' + key + ': ' + str(settings[key]))

# ---------------------------------------- READING ----------
complex_part = 'phase'                                         # 'abs' or 'phase' will reading
groups = {                                                     # Use regex. Only exist groups will be added
    '.*itch.*': 'kitchen',
    'room.*': 'room',
    '.*bathroom.*': 'bathroom',
    'hall.*': 'hall',
    'toilet.*': 'toilet',
    '.*air.*': 'air',
    '.*bottle.*': 'bottle',
    '.*thermos.*': 'thermos',
    '.*grater.*': 'grater',
    '.*casserole.*': 'casserole',
    '.*dish.*': 'dish',
}

main_path = path.join('csi', 'use_in_paper', '4_objects')
# main_path = path.join('csi', 'homelocation', 'two place')
train_path = path.join(main_path, 'train')
test_path = path.join(main_path, 'test')
 
df_train = prep.concat_csi(readcsi.get_csi_dfs(train_path, groups, complex_part))
df_test = prep.concat_csi(readcsi.get_csi_dfs(test_path, groups, complex_part))

print('Train packets number:\t', df_train.shape[0], 'Packets:', df_train['object_type'].unique())
print('Test packets number:\t', df_test.shape[0], 'Packets:', df_test['object_type'].unique())
print('Reading complete -->', round(time() - time_start, 2))

# ---------------------------------------- PREPARATION ----------
if settings['normalize']:
    if complex_part == 'phase':
        df_train = prep.concat_csi(prep.normalize_phase(*prep.split_csi(df_train)))
        df_test = prep.concat_csi(prep.normalize_phase(*prep.split_csi(df_test)))
    else:
      print('Can`t normalize abs! This option only for phase')

if settings['make_smooth']:
    window = 6  # Smoothing window width
    win_type = 'hamming'
    df_train = prep.concat_csi(prep.smooth(*prep.split_csi(df_train), window=window, win_type=win_type))
    df_test = prep.concat_csi(prep.smooth(*prep.split_csi(df_test), window=window, win_type=win_type))

if settings['make_reduce']:
    df_train = prep.decimate_one(df_train, 5, 7, 9, 11, 13)
    print('New df_train size:', df_train.shape[0])

if settings['make_same']:
    df_train = prep.make_same(df_train)
    df_test = prep.make_same(df_test)

for _ in range(settings['diff_order']):
    df_train = prep.concat_csi(prep.difference(*prep.split_csi(df_train)))
    df_test = prep.concat_csi(prep.difference(*prep.split_csi(df_test)))
    plot.csi_plot_types(df_train.head(15))

# ---------------------------------------- CLASSIFICATION ----------
clf_res = pd.concat([
    # ml.fit_ffnn(df_train, df_test),
    # ml.fit_cnn(df_train, df_test),
    ml.fit_sklearn(df_train, df_test),
    ])

# ---------------------------------------- RESULTS COMPARISON ----------
sorted_res = clf_res.sort_values(by='method name', ascending=True, ignore_index=True)

print('\nClassification results:\n', sorted_res)
sorted_res.to_csv('results\\ml_results.csv', index=False)

print('Finish -->', round(time() - time_start, 2))