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

#using gridsearch find best hyperparameters of RandomForestClassifier
def grid_search_randomforest(xTrain, xTest, yTrain, yTest,type):
    diparameter={"n_estimators":[i for i in range(100,500,1501,500)],"max_depth":[i for i in range(10,31,10)],"criterion":["gini","entropy"]}
    licv=GridSearchCV(RandomForestClassifier(),param_grid=diparameter,cv=10,n_jobs=None)
    licv.fit(xTrain,yTrain)
    predictor=licv.best_estimator_
    joblib.dump(predictor,"randomforest_"+type+".pkl",compress=True)
    score=licv.score(xTest,yTest)
    print(score)
    print(licv.best_estimator_)
    return licv

#using gridsearch find best hyperparameters of DecisionTreeClassifier
def grid_search_decisiontree(xTrain, xTest, yTrain, yTest,type):
    diparameter={"max_depth":[i for i in range(10,31,10)],"criterion":["gini","entropy"]}
    licv=GridSearchCV(DecisionTreeClassifier(),param_grid=diparameter,cv=10,n_jobs=None)
    licv.fit(xTrain,yTrain)
    predictor=licv.best_estimator_
    joblib.dump(predictor,"decisiontree_"+type+".pkl",compress=True)
    score=licv.score(xTest,yTest)
    print(score)
    print(licv.best_estimator_)
    return licv

#feature selection by sklearn randomforest and SelectFromModel
def feature_select_randomforest(xTrain,yTrain, xTest, threshold=0.005):
    clf=RandomForestClassifier(n_estimators=100,max_depth=20,criterion="entropy")
    sfm = SelectFromModel(clf, threshold)
    sfm.fit(xTrain, yTrain)
    X_important_train = sfm.transform(xTrain)
    X_important_test = sfm.transform(xTest)
    return X_important_train,X_important_test


#feature selection by PCA
def feature_select_pca(xTrain):
    scaler = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scaler.fit_transform(xTrain)
    pca = PCA().fit(data_rescaled)
    #Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Find best components of PCA')
    plt.savefig("PCA_components.png")
    plt.show()
 

def final_classifier(xTrain, xTest, yTrain, yTest):
    clf4=RandomForestClassifier(n_estimators=100,max_depth=20,criterion="entropy")
    clf5=DecisionTreeClassifier(max_depth=20,criterion="entropy")
    clf6=GradientBoostingClassifier()
    clf4.fit(xTrain,yTrain)
    clf5.fit(xTrain,yTrain)
    clf6.fit(xTrain,yTrain)
    print(clf4.score(xTest,yTest))    
    print(clf5.score(xTest,yTest))    
    print(clf6.score(xTest,yTest))    
    
    X_important_train,X_important_test=feature_select_randomforest(xTrain,yTrain, xTest)
    clf1=RandomForestClassifier(n_estimators=100,max_depth=20,criterion="entropy")
    clf2=DecisionTreeClassifier(max_depth=20,criterion="entropy")
    clf3=GradientBoostingClassifier()
    clf4.fit(X_important_train,yTrain)
    clf5.fit(X_important_train,yTrain)
    clf6.fit(X_important_train,yTrain)
    print(clf4.score(X_important_test,yTest))    
    print(clf5.score(X_important_test,yTest))    
    print(clf6.score(X_important_test,yTest))    
    eclf = VotingClassifier(estimators=[
        ('rfc', clf1), ('dtc', clf2), ('gbc', clf3)],
        voting='hard',
        flatten_transform=True)
    eclf.fit(X_important_train,yTrain)
    print(eclf.score(X_important_test,yTest))    


def final_classifier_pca(xTrain, xTest, yTrain, yTest):
    pca = PCA(n_components=50)
    scaler = MinMaxScaler(feature_range=[0, 1])
    data_rescaled_train = scaler.fit_transform(xTrain)
    X_important_train = pca.fit_transform(data_rescaled_train)
    data_rescaled_test = scaler.fit_transform(xTest)
    X_important_test = pca.transform(data_rescaled_test)

    clf1=RandomForestClassifier(n_estimators=100,max_depth=20,criterion="entropy")
    clf2=DecisionTreeClassifier(max_depth=20,criterion="entropy")
    clf3=GradientBoostingClassifier()
    clf4=RandomForestClassifier(n_estimators=100,max_depth=20,criterion="entropy")
    clf5=DecisionTreeClassifier(max_depth=20,criterion="entropy")
    clf6=GradientBoostingClassifier()
    clf4.fit(X_important_train,yTrain)
    clf5.fit(X_important_train,yTrain)
    clf6.fit(X_important_train,yTrain)
    print(clf4.score(X_important_test,yTest))    
    print(clf5.score(X_important_test,yTest))    
    print(clf6.score(X_important_test,yTest))    
    eclf = VotingClassifier(estimators=[
        ('rfc', clf1), ('dtc', clf2), ('gbc', clf3)],
        voting='hard',
        flatten_transform=True)
    eclf.fit(X_important_train,yTrain)
    print(eclf.score(X_important_test,yTest))    
