from __future__ import annotations
from ..watcher import Watcher as W
import pandas as pd
import numpy as np
import scipy


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from time import time


# Также в этом классе будут функции для статистического анализа и построения графиков

# Класс для методов машинного обучения
class WifiLearn:
    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.lens = { 'train': x_train.shape[0], 'test': x_test.shape[0] }
        self.__w = W()
        self.__w.hprint(self.__w.INFO, 'WifiLearn: create with ' + str(self.lens['train']) + ' train and ' + str(self.lens['test']) + ' test packets')
        self.results = []


    def normalize(self):
        pass


    def shuffle(self, part: int=1):
        pass


    def print(self) -> WifiLearn:
        print(pd.DataFrame(self.results))
        return self


    @W.stopwatch
    def fit_classic(self) -> WifiLearn:
        self.__w.hprint(self.__w.INFO, 'WifiLearn: start fit_classic')
        classifiers = {  # You can add your clfs or change params here:
            'Logistic Regression':              LogisticRegression(max_iter=10000),
            'SVC':                              SVC(),
            'K-nearest neighbors':              KNeighborsClassifier(),
            'Gaussian Naive Bayes':             GaussianNB(),
            'Perceptron':                       Perceptron(),
            'Linear SVC':                       LinearSVC(max_iter=10000),
            'Stochastic Gradient Descent':      SGDClassifier(),
            'Random Forest':                    RandomForestClassifier(max_depth=20),
            'sk-learn Neural Net':              MLPClassifier(hidden_layer_sizes=(200, 20)),
            'Ada Boost':                        AdaBoostClassifier()
        }

        res = []

        for clf in classifiers:
            start_fit = time()
            classifiers[clf].fit(self.x_train, self.y_train)
            res.append({'name': clf, 'accuracy': round(classifiers[clf].score(self.x_test, self.y_test) * 100, 2),'duration': round(time() - start_fit, 2)})
            self.__w.hprint(self.__w.BOLD, 'WifiLearn: fit ' + clf + ': ' + str(res[-1]['accuracy']))

        self.results += res
        return self
