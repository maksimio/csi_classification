# This is the main module
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

print('Imports complete -->', round(time() - time_start, 2))

# ---------- DATA READING ----------
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
print('Preparation complete -->', round(time() - time_start, 2))

# ---------- VISUALISATION ----------


# ---------- DATA PREPARATION ----------
x_train = df_train.drop('object_type',axis=1)
y_train = df_train['object_type']
x_test = df_test.drop('object_type',axis=1)
y_test = df_test['object_type']

# ---------- CLASSIFICATION ----------
clf_res = pd.DataFrame(columns=('method name','accuracy'))

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Logistic Regression', round(logreg.score(x_test, y_test) * 100, 2)]
print('Logistic Regression -->', round(time() - time_start, 2))

svc = SVC()
svc.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Support Vector Machines', round(svc.score(x_test, y_test) * 100, 2)]
print('Support Vector Machines -->', round(time() - time_start, 2))

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['K-nearest neighbors', round(knn.score(x_test, y_test) * 100, 2)]
print('K-nearest neighbors -->', round(time() - time_start, 2))

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Gaussian Naive Bayes', round(gaussian.score(x_test, y_test) * 100, 2)]
print('Gaussian Naive Bayes -->', round(time() - time_start, 2))

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Perceptron', round(perceptron.score(x_test, y_test) * 100, 2)]
print('Perceptron -->', round(time() - time_start, 2))

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Linear SVC', round(linear_svc.score(x_test, y_test) * 100, 2)]
print('Linear SVC -->', round(time() - time_start, 2))

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Stochastic Gradient Descent', round(sgd.score(x_test, y_test) * 100, 2)]
print('Stochastic Gradient Descent -->', round(time() - time_start, 2))

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Decision Tree', round(decision_tree.score(x_test, y_test) * 100, 2)]
print('Decision Tree -->', round(time() - time_start, 2))

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
clf_res.loc[len(clf_res)] = ['Random Forest', round(random_forest.score(x_test, y_test) * 100, 2)]
print('Random Forest -->', round(time() - time_start, 2))


# ---------- RESULTS COMPARISON ----------
print(clf_res)