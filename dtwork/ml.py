'''Machine-learning module.'''
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import keras
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from time import time
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt

def keras_prepare(df_train, df_test):
    x_train = df_train.drop('object_type', axis=1).to_numpy()
    x_test = df_test.drop('object_type', axis=1).to_numpy()
    y_train = df_train['object_type'].to_numpy()
    y_test = df_test['object_type'].to_numpy()

    # Convert to categorical:
    obj_lst = sorted(df_train['object_type'].unique())
    i = 0
    for o_name in obj_lst:
        y_train[y_train == o_name] = i
        y_test[y_test == o_name] = i
        i += 1
    y_train = utils.to_categorical(y_train, len(obj_lst))
    y_test = utils.to_categorical(y_test, len(obj_lst))

    return x_train, x_test, y_train, y_test


def fit_sklearn(df_train, df_test):
    '''This function of Machine learning use simple models for fit from sk-learn'''
    x_train = df_train.drop('object_type', axis=1)
    y_train = df_train['object_type']
    x_test = df_test.drop('object_type', axis=1)
    y_test = df_test['object_type']

    classifiers = {  # You can add your clfs or change params here:
        'Logistic Regression':              LogisticRegression(max_iter=10000),
        'SVC':                              SVC(),
        'K-nearest neighbors':              KNeighborsClassifier(),
        'Gaussian Naive Bayes':             GaussianNB(),
        'Perceptron':                       Perceptron(),
        'Linear SVC':                       LinearSVC(max_iter=10000),
        'Stochastic Gradient Descent':      SGDClassifier(),
        'Decision Tree':                    DecisionTreeClassifier(max_depth=20),
        'Random Forest':                    RandomForestClassifier(max_depth=20),
        'sk-learn Neural Net':              MLPClassifier(hidden_layer_sizes=(200, 20)),
        'Ada Boost':                        AdaBoostClassifier()
    }

    clf_res = pd.DataFrame(columns=('method name', 'accuracy', 'time'))

    for clf in classifiers:
        start_fit = time()
        classifiers[clf].fit(x_train, y_train)
        clf_res.loc[len(clf_res)] = [clf, round(classifiers[clf].score(x_test, y_test) * 100, 2), round(time() - start_fit, 2)]
        print(clf, '-->', round(time(), 2))

    return clf_res


def fit_ffnn(df_train, df_test):
    x_train, x_test, y_train, y_test = keras_prepare(df_train, df_test)

    model = Sequential()
    model.add(Dense(360, input_dim=x_train.shape[1], activation='hard_sigmoid'))
    model.add(Dense(len(df_train['object_type'].unique()), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam', metrics=['accuracy'])

    start_fit = time()
    model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=0, validation_split=0.1)
    score = round(model.evaluate(x_test, y_test)[1] * 100, 2)
    # clf_res.loc[len(clf_res)] = ['keras FFNN', score,
    #                              round(time() - start_fit, 2)]

    print('keras FFNN -->', round(time(), 2))
    print('keras record:', )


def fit_cnn(df_train, df_test):
    x_train, x_test, y_train, y_test = keras_prepare(df_train, df_test)

    batch_size = 50
    nb_epoch = 50
    x_train = np.reshape(x_train, (-1, 4, 56, 1)) / 400
    x_test = np.reshape(x_test, (-1, 4, 56, 1)) / 400

    model = Sequential()
    model.add(Conv2D(28, (3, 3), padding='same',input_shape=(4, 56, 1), activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(12, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(len(df_train['object_type'].unique()), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.1, shuffle=True, verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
    #print(model.summary())
    #model.save('results\\cnn2')


def cnn_load(df_train, df_test):
    # Convert to numpy:
    x_train_net = df_train.drop('object_type', axis=1).to_numpy()
    x_test_net = df_test.drop('object_type', axis=1).to_numpy()
    y_train_net = df_train['object_type'].to_numpy()
    y_test_net = df_test['object_type'].to_numpy()

    x_train = np.reshape(x_train_net, (-1, 4, 56, 1))
    x_test = np.reshape(x_test_net, (-1, 4, 56, 1))

    # batch_size = 50
    # nb_epoch = 50

    # # Нормализуем данные
    # x_train /= 400
    # x_test /= 400

    # obj_lst = sorted(df_train['object_type'].unique())
    # i = 0
    # for o_name in obj_lst:
    #     y_train_net[y_train_net == o_name] = i
    #     y_test_net[y_test_net == o_name] = i
    #     i += 1
    # y_train = utils.to_categorical(y_train_net, len(obj_lst))
    # y_test = utils.to_categorical(y_test_net, len(obj_lst))

    # model = keras.models.load_model('results\\cnn')
    # scores = model.evaluate(x_test, y_test, verbose=0)
    # print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
    # print(model.summary())

    # activ_model = keras.Model(
    #     inputs=model.input, outputs=model.layers[4].output)
    # activation = activ_model.predict(x_train)
    # print(activation.shape)

    # plt.matshow(activation[0, :, :, 0], cmap='viridis')
    # plt.matshow(activation[1, :, :, 0], cmap='viridis')
    # plt.matshow(activation[2, :, :, 0], cmap='viridis')
    # plt.matshow(activation[3, :, :, 0], cmap='viridis')

    # plt.show()
