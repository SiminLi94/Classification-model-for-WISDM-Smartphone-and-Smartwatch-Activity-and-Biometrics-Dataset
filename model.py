from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression,CCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os

class Model_RF_PCA:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100,max_depth=20,criterion="entropy")
        self.pca=PCA(n_components=50)
        self.scaler = MinMaxScaler(feature_range=[0, 1])
        
    def train(self, X_train, y_train):
        
        data_rescaled = self.scaler.fit_transform(X_train)
        X_important = self.pca.fit_transform(data_rescaled)
        
        self.rf.fit(X_important, y_train)

    def predict(self, X_test):
        data_rescaled = self.scaler.transform(X_test)
        X_important = self.pca.fit_transform(data_rescaled)

        pred = self.rf.predict(X_important)
        return pred
    def score(self, X_test,y_test):
        data_rescaled = self.scaler.transform(X_test)
        X_important = self.pca.transform(data_rescaled)
        score = self.rf.score(X_important,y_test)
        return score