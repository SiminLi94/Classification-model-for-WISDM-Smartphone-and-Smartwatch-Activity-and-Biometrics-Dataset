import sys
sys.path.append("..")
sys.path.append(".")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from data_process import get_data_from_arff,set_data_index,get_data_from_df
from sklearn.model_selection import GridSearchCV
from performance_process import feature_select_pca,grid_search_decisiontree,grid_search_randomforest,final_classifier_pca,final_classifier
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression,CCA
from model import Model_RF_PCA

#from model_random_forest import RandomForestM

data_type=[['phone', 'accel'], ['phone', 'gyro'],['watch', 'accel'], ['watch', 'gyro']]

df_phone_accel=set_data_index(get_data_from_arff(data_type[0]))
df_phone_gyro=set_data_index(get_data_from_arff(data_type[1]))
df_watch_accel=set_data_index(get_data_from_arff(data_type[2]))
df_watch_gyro=set_data_index(get_data_from_arff(data_type[3]))

df_phone = pd.merge(df_phone_accel, df_phone_gyro, on=['level_1','"ACTIVITY"','"class"'])
df_watch = pd.merge(df_watch_accel, df_watch_gyro, on=['level_1','"ACTIVITY"','"class"'])
df_accel = pd.merge(df_phone_accel, df_watch_accel, on=['level_1','"ACTIVITY"','"class"'])
df_gyro = pd.merge(df_phone_gyro, df_watch_gyro, on=['level_1','"ACTIVITY"','"class"'])

df_all=pd.merge(df_phone, df_watch, on=['level_1','"ACTIVITY"','"class"'])

#all featrurs(phone, watch, accel, gyro)
xTrain, xTest, yTrain, yTest = get_data_from_df(df_all)

clf=Model_RF_PCA()
clf.train(xTrain,yTrain)
score=clf.score(xTest,yTest)
#
#final_classifier(xTrain, xTest, yTrain, yTest)
#feature_select_pca(xTrain)
#final_classifier_pca(xTrain, xTest, yTrain, yTest)


#watch featrurs(watch, accel, gyro)
#xTrain, xTest, yTrain, yTest = get_data_from_df(df_watch)
#grid_search_decisiontree(xTrain, xTest, yTrain, yTest,'watch')
#grid_search_randomforest(xTrain, xTest, yTrain, yTest,'watch')
#feature_select_cca(xTrain,yTrain)

##phone featrurs(phone, watch, accel, gyro)
#xTrain, xTest, yTrain, yTest = get_data_from_df(df_phone)
#randomforest(xTrain, xTest, yTrain, yTest,'phone',df_phone.columns)
#
##phone featrurs(phone, watch, accel, gyro)
#xTrain, xTest, yTrain, yTest = get_data_from_df(df_accel)
#randomforest(xTrain, xTest, yTrain, yTest,'accel',df_accel.columns)
#
##phone featrurs(phone, watch, accel, gyro)
#xTrain, xTest, yTrain, yTest = get_data_from_df(df_gyro)
#randomforest(xTrain, xTest, yTrain, yTest,'gyro',df_gyro.columns)
#
##phone featrurs(phone, watch, accel, gyro)
#xTrain, xTest, yTrain, yTest = get_data_from_df(df_phone_accel)
#randomforest(xTrain, xTest, yTrain, yTest,'phone_accel',df_phone_accel.columns)
#
##phone featrurs(phone, watch, accel, gyro)
#xTrain, xTest, yTrain, yTest = get_data_from_df(df_phone_gyro)
#randomforest(xTrain, xTest, yTrain, yTest,'phone_gyro',df_phone_gyro.columns)
#
##phone featrurs(phone, watch, accel, gyro)
#xTrain, xTest, yTrain, yTest = get_data_from_df(df_watch_accel)
#randomforest(xTrain, xTest, yTrain, yTest,'watch_accel',df_watch_accel.columns)
#
##phone featrurs(phone, watch, accel, gyro)
#xTrain, xTest, yTrain, yTest = get_data_from_df(df_watch_gyro)
#randomforest(xTrain, xTest, yTrain, yTest,'watch_gyro',df_watch_gyro.columns)
