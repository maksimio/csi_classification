from time import time
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Our backend is TensorFlow
from keras.models import Sequential
from keras.layers import Dense
from keras import utils
import keras

from matplotlib import pyplot as plt


def ml(x_train, y_train, x_test, y_test, df_train, df_test, time_start, use_keras=True):
  '''The main machine learning function'''    
  classifiers = { # You can add your clfs or change params here:
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'SVC': SVC(),
    'K-nearest neighbors': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Perceptron': Perceptron(),
    'Linear SVC': LinearSVC(max_iter=10000),
    'Stochastic Gradient Descent': SGDClassifier(),
    'Decision Tree': DecisionTreeClassifier(max_depth=20),
    'Random Forest': RandomForestClassifier(max_depth=20),
    'sk-learn Neural Net': MLPClassifier(hidden_layer_sizes=(200, 20)),
    'Ada Boost': AdaBoostClassifier(),
  }

  clf_res = pd.DataFrame(columns=('method name', 'accuracy', 'time'))

  for clf in classifiers:
    start_fit = time()
    classifiers[clf].fit(x_train, y_train)
    clf_res.loc[len(clf_res)] = [clf, round(classifiers[clf].score(x_test, y_test) * 100, 2), round(time() - start_fit, 2)]
    print(clf, '-->', round(time() - time_start, 2))

  # ---------- FFNN ----------
  if use_keras:
    # Convert to numpy:
    x_train_net = df_train.drop('object_type', axis=1).to_numpy()
    x_test_net = df_test.drop('object_type', axis=1).to_numpy()
    y_train_net = df_train['object_type'].to_numpy()
    y_test_net = df_test['object_type'].to_numpy()
    
    # Convert to categorical:
    obj_lst = sorted(df_train['object_type'].unique())
    i = 0
    for o_name in obj_lst:
      y_train_net[y_train_net == o_name] = i
      y_test_net[y_test_net == o_name] = i
      i += 1
    y_train_net = utils.to_categorical(y_train_net, len(obj_lst))
    y_test_net = utils.to_categorical(y_test_net, len(obj_lst))

    # FFNN - Feed Forward neural network:
    model = Sequential()
    model.add(Dense(360, input_dim=x_train_net.shape[1], activation='hard_sigmoid'))
    model.add(Dense(len(obj_lst), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

    start_fit = time()
    model.fit(x_train_net, y_train_net, batch_size=200, epochs=100, verbose=0, validation_split=0.1)
    score = round(model.evaluate(x_test_net, y_test_net)[1] * 100, 2)
    clf_res.loc[len(clf_res)] = ['keras FFNN', score, round(time() - start_fit, 2)]

    print('keras FFNN -->', round(time() - time_start, 2))
  return clf_res


import numpy
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

# Задаем seed для повторяемости результатов
numpy.random.seed(42)

def cnn(df_train, df_test):
    # Convert to numpy:
    x_train_net = df_train.drop('object_type', axis=1).to_numpy()
    x_test_net = df_test.drop('object_type', axis=1).to_numpy()
    y_train_net = df_train['object_type'].to_numpy()
    y_test_net = df_test['object_type'].to_numpy()

    x_train = np.reshape(x_train_net, (-1, 4, 56, 1))
    x_test = np.reshape(x_test_net, (-1, 4, 56, 1))

    batch_size = 50
    nb_epoch = 50

    # Нормализуем данные
    x_train /= 400
    x_test /= 400

    obj_lst = sorted(df_train['object_type'].unique())
    i = 0
    for o_name in obj_lst:
      y_train_net[y_train_net == o_name] = i
      y_test_net[y_test_net == o_name] = i
      i += 1
    y_train = utils.to_categorical(y_train_net, len(obj_lst))
    y_test = utils.to_categorical(y_test_net, len(obj_lst))
    #print(x_train)

    # Создаем последовательную модель
    model = Sequential()
    model.add(Conv2D(28, (3, 3), padding='same', input_shape=(4, 56, 1), activation='relu'))
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
    model.add(Dense(len(obj_lst), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.1, shuffle=True, verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
    print(model.summary())
    model.save('results\\cnn2')




def cnn_load(df_train, df_test):
    # Convert to numpy:
    x_train_net = df_train.drop('object_type', axis=1).to_numpy()
    x_test_net = df_test.drop('object_type', axis=1).to_numpy()
    y_train_net = df_train['object_type'].to_numpy()
    y_test_net = df_test['object_type'].to_numpy()

    x_train = np.reshape(x_train_net, (-1, 4, 56, 1))
    x_test = np.reshape(x_test_net, (-1, 4, 56, 1))

    batch_size = 50
    nb_epoch = 50

    # Нормализуем данные
    x_train /= 400
    x_test /= 400

    obj_lst = sorted(df_train['object_type'].unique())
    i = 0
    for o_name in obj_lst:
      y_train_net[y_train_net == o_name] = i
      y_test_net[y_test_net == o_name] = i
      i += 1
    y_train = utils.to_categorical(y_train_net, len(obj_lst))
    y_test = utils.to_categorical(y_test_net, len(obj_lst))

    model = keras.models.load_model('results\\cnn')
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
    print(model.summary())  

    activ_model = keras.Model(inputs=model.input, outputs=model.layers[4].output)
    activation = activ_model.predict(x_train)
    print(activation.shape)

    plt.matshow(activation[0, :, :, 0], cmap='viridis')
    plt.matshow(activation[1, :, :, 0], cmap='viridis')
    plt.matshow(activation[2, :, :, 0], cmap='viridis')
    plt.matshow(activation[3, :, :, 0], cmap='viridis')

    plt.show()

    





































def cnn2(df_train, df_test):
    # Convert to numpy:
    x_train_net = df_train.drop('object_type', axis=1).to_numpy()
    x_test_net = df_test.drop('object_type', axis=1).to_numpy()
    y_train_net = df_train['object_type'].to_numpy()
    y_test_net = df_test['object_type'].to_numpy()

    x_train = np.reshape(x_train_net, (-1, 4, 56, 1))
    x_test = np.reshape(x_test_net, (-1, 4, 56, 1))

    batch_size = 50
    nb_epoch = 50

    # Нормализуем данные
    x_train /= 400
    x_test /= 400

    obj_lst = sorted(df_train['object_type'].unique())
    i = 0
    for o_name in obj_lst:
      y_train_net[y_train_net == o_name] = i
      y_test_net[y_test_net == o_name] = i
      i += 1
    y_train = utils.to_categorical(y_train_net, len(obj_lst))
    y_test = utils.to_categorical(y_test_net, len(obj_lst))
    #print(x_train)

    # Создаем последовательную модель
    model = Sequential()
    model.add(Conv2D(14, (3, 3), padding='same', input_shape=(4, 56, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.33))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.33))

    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(obj_lst), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.1, shuffle=True, verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
    print(model.summary())