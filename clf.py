# This is the main modul
# ---------- MODULE IMPORTS ----------
from time import time
time_start = time()
import pandas as pd
from os import path
from dtwork import readcsi

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

print('Imports complete -->', round(time() - time_start, 3), '\n')

# ---------- DATA PREPARATION ----------
groups = {
    ".*bottle.*": "bottle",
    ".*air.*": "air"
}
main_path = path.join("csi", "use_in_paper","2_objects")
train_path = path.join(main_path,"train")
test_path = path.join(main_path, "test")

df_train = readcsi.get_abs_csi_df_big(train_path,groups)
df_test = readcsi.get_abs_csi_df_big(test_path,groups)

print('Train packets number:\t', df_train.shape[0])
print('Test packets number:\t', df_test.shape[0])
print('Preparation complete -->', round(time() - time_start, 3), '\n')

# ---------- VISUALISATION ----------


# ---------- CLASSIFICATION ----------





# ---------- DATA PREPARATION ----------
# ---------- DATA PREPARATION ----------
# ---------- DATA PREPARATION ----------
