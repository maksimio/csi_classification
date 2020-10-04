# This is the main module
use_keras = True
plot_and_exit = False
select_features = True
# ---------- MODULE IMPORTS ----------
from time import time
time_start = time()
import pandas as pd
from os import path
from dtwork import readcsi
from dtwork import prep
from dtwork import plot

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

if use_keras: # Our backend is TensorFlow
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import utils

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
df_train = df_train.sample(frac=1).reset_index(drop=True) # Mix df
df_test = df_test.sample(frac=1).reset_index(drop=True)

x_train = df_train.drop('object_type',axis=1)
y_train = df_train['object_type']
x_test = df_test.drop('object_type',axis=1)
y_test = df_test['object_type']

# ---------- CLASSIFICATION (sklearn) ----------
# Feature selection with statistic
kBest = SelectKBest(score_func=chi2, k=10)
scores_train = kBest.fit(x_train, y_train).scores_ # depend of df size
scores_test = kBest.fit(x_test, y_test).scores_
df = pd.DataFrame(pd.Series(scores_train,name='train'),pd.Series(x_train.columns,name='subcarriers'))
df['train_%'] = (df['train']/df['train'].max()*100).astype(int)
df['test'] = pd.Series(scores_test,name='test')
df['test_%'] = (df['test']/df['test'].max()*100).astype(int)
df['sig_way'] = [i//56+1 for i in range(224)]
df['subc_num'] = [i % 56 + 1 for i in range(224)]
df['test'] = df['test'].astype(int)
df['train'] = df['train'].astype(int)

df.to_csv('results\\statistic_correlation.csv') # In this file you can see correlation
# for train and test datasets for all 4 (in our case) ways of signal between antennas





exit()

clf_res = pd.DataFrame(columns=('method name','accuracy','time'))



logreg = LogisticRegression(max_iter=10000)
start_fit = time()
logreg.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Logistic Regression', round(logreg.score(x_test, y_test) * 100, 2), round(time()-start_fit,2)]
print('Logistic Regression -->', round(time() - time_start, 2))

svc = SVC()
start_fit = time()
svc.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Support Vector Machines', round(svc.score(x_test, y_test) * 100, 2), round(time()-start_fit,2)]
print('Support Vector Machines -->', round(time() - time_start, 2))

knn = KNeighborsClassifier()
start_fit = time()
knn.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['K-nearest neighbors', round(knn.score(x_test, y_test) * 100, 2), round(time()-start_fit,2)]
print('K-nearest neighbors -->', round(time() - time_start, 2))

gaussian = GaussianNB()
start_fit = time()
gaussian.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Gaussian Naive Bayes', round(gaussian.score(x_test, y_test) * 100, 2), round(time()-start_fit,2)]
print('Gaussian Naive Bayes -->', round(time() - time_start, 2))

perceptron = Perceptron()
start_fit = time()
perceptron.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Perceptron', round(perceptron.score(x_test, y_test) * 100, 2), round(time()-start_fit,2)]
print('Perceptron -->', round(time() - time_start, 2))

linear_svc = LinearSVC(max_iter=10000)
start_fit = time()
linear_svc.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Linear SVC', round(linear_svc.score(x_test, y_test) * 100, 2), round(time()-start_fit,2)]
print('Linear SVC -->', round(time() - time_start, 2))

sgd = SGDClassifier()
start_fit = time()
sgd.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Stochastic Gradient Descent', round(sgd.score(x_test, y_test) * 100, 2), round(time()-start_fit,2)]
print('Stochastic Gradient Descent -->', round(time() - time_start, 2))

decision_tree = DecisionTreeClassifier()
start_fit = time()
decision_tree.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Decision Tree', round(decision_tree.score(x_test, y_test) * 100, 2), round(time()-start_fit,2)]
print('Decision Tree -->', round(time() - time_start, 2))

random_forest = RandomForestClassifier()
start_fit = time()
random_forest.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Random Forest', round(random_forest.score(x_test, y_test) * 100, 2), round(time()-start_fit,2)]
print('Random Forest -->', round(time() - time_start, 2))

# ---------- FFNN ----------
if use_keras:
    # Convert to numpy:
    x_test = df_test.drop('object_type', axis=1).to_numpy()
    x_train = df_train.drop('object_type', axis=1).to_numpy()
    y_test = df_test['object_type'].to_numpy()
    y_train = df_train['object_type'].to_numpy()

    # Convert to categorical:
    obj_lst = sorted(df_train['object_type'].unique())
    i = 0
    for o_name in obj_lst:
        y_train[y_train == o_name] = i
        y_test[y_test == o_name] = i
        i += 1
    y_train = utils.to_categorical(y_train,len(obj_lst))
    y_test = utils.to_categorical(y_test,len(obj_lst))

    print('Finish convert -->', round(time() - time_start, 2))

    # FFNN:
    model3 = Sequential()
    model3.add(Dense(360, input_dim=224, activation="hard_sigmoid"))
    model3.add(Dense(2, activation="softmax"))
    model3.compile(loss="categorical_crossentropy", optimizer="Nadam", metrics=["accuracy"])
    print(model3.summary())
    start_fit = time()
    model3.fit(x_train, y_train, batch_size=200, epochs=100, verbose=0, validation_split=0.1)
    score = round(model3.evaluate(x_test, y_test)[1]*100,2)
    clf_res.loc[len(clf_res)] = ['FFNN', score, round(time()-start_fit,2)]

    print('FFNN -->', round(time() - time_start, 2))

# ---------- SELECTION BEST FEATURES ----------
if select_features:
    top_fnum = 20 # Number of top selected features
    
    # 1.Use Univariate Statistical Tests
    sel = SelectKBest(score_func=chi2, k=top_fnum).fit(x_train, y_train)
    features = x_train.columns[sel.get_support()]
    print('Statistical selection:',features)




# ---------- RESULTS COMPARISON ----------
sorted_res = clf_res.sort_values('accuracy',ascending=False)
print('Classification results:')
print(sorted_res)
print('Finish -->', round(time() - time_start, 2))