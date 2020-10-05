from time import time
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Our backend is TensorFlow
from keras.models import Sequential
from keras.layers import Dense
from keras import utils


def ml(x_train, y_train, x_test, y_test, df_train, df_test, time_start, use_keras=True):
    '''The main machine learning function'''
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
        x_test_net = df_test.drop('object_type', axis=1).to_numpy()
        x_train_net = df_train.drop('object_type', axis=1).to_numpy()
        y_test_net = df_test['object_type'].to_numpy()
        y_train_net = df_train['object_type'].to_numpy()

        # Convert to categorical:
        obj_lst = sorted(df_train['object_type'].unique())
        i = 0
        for o_name in obj_lst:
            y_train_net[y_train_net == o_name] = i
            y_test_net[y_test_net == o_name] = i
            i += 1
        y_train_net = utils.to_categorical(y_train_net,len(obj_lst))
        y_test_net = utils.to_categorical(y_test_net,len(obj_lst))

        print('Finish convert -->', round(time() - time_start, 2))

        # FFNN:
        model3 = Sequential()
        model3.add(Dense(360, input_dim=x_train_net.shape[1], activation="hard_sigmoid"))
        model3.add(Dense(2, activation="softmax"))
        model3.compile(loss="categorical_crossentropy", optimizer="Nadam", metrics=["accuracy"])
        print(model3.summary())
        start_fit = time()
        model3.fit(x_train_net, y_train_net, batch_size=200, epochs=100, verbose=0, validation_split=0.1)
        score = round(model3.evaluate(x_test_net, y_test_net)[1]*100,2)
        clf_res.loc[len(clf_res)] = ['FFNN', score, round(time()-start_fit,2)]

        print('FFNN -->', round(time() - time_start, 2))
    
    return clf_res