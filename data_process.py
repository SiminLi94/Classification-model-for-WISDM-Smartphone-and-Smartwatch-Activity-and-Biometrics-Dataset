
import numpy as np
import os
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#read data files
def get_data_from_arff(row_path):
#    ROOT='C:\\@@@@BISHOPS\\CS507\\FIANAL PROJECT\\arff_files\\'
    ROOT='.\\arff_files\\'
    p=os.path.join(os.path.join(ROOT,row_path[0]),row_path[1])
    df = pd.DataFrame()
    for b in range(0,51):
        f = os.path.join(p, 'data_%d_%s_%s.arff' % (1600+b,row_path[1],row_path[0] ))
        if os.path.exists(f):
            print(f)
            data, meta = arff.loadarff(f)
            df=df.append(pd.DataFrame(data),ignore_index=True)
    df.columns = meta.names()

    return df

#set index to data according to activity and class
def set_data_index(df):
    grouped = df.groupby(['"ACTIVITY"','"class"'],as_index = False)
    grouped_index = grouped.apply(lambda x: x.reset_index(drop = True)).reset_index()
    df_with_idx=grouped_index.drop('level_0',axis = 1).set_index('level_1')
    return df_with_idx
 
#transfer datafram to xtrain xtest ytrain ytest
def get_data_from_df(df):
    df=df.drop(columns=['"class"'])
    data_all=df.values
    y_activity=data_all[:,0]
    y_activity[y_activity==b'A'] = 1
    y_activity[y_activity==b'B'] = 2
    y_activity[y_activity==b'C'] = 3
    y_activity[y_activity==b'D'] = 4
    y_activity[y_activity==b'E'] = 5
    y_activity[y_activity==b'F'] = 6
    y_activity[y_activity==b'G'] = 7
    y_activity[y_activity==b'H'] = 8
    y_activity[y_activity==b'I'] = 9
    y_activity[y_activity==b'J'] = 10
    y_activity[y_activity==b'K'] = 11
    y_activity[y_activity==b'L'] = 12
    y_activity[y_activity==b'M'] = 13
    y_activity[y_activity==b'O'] = 14
    y_activity[y_activity==b'P'] = 15
    y_activity[y_activity==b'Q'] = 16
    y_activity[y_activity==b'R'] = 17
    y_activity[y_activity==b'S'] = 18
    
    X_data=data_all[:,1:].astype(None)        
    
    data_Xy=np.hstack((X_data,y_activity.reshape(X_data.shape[0],1)))
    
    corrmat=np.corrcoef(data_Xy.astype(None).T)
    plt.figure(figsize=(8,8))
    plt.imshow(corrmat)
    plt.title("Correlation of variances")
    plt.savefig("Correlation.png")
#    plt.plot(corrmat[:,-1],'o')
    
    plt.show()  
    
    np.random.shuffle(data_Xy)
    X_shuf=data_Xy[:,:-1]
    y_shuf_activity=data_Xy[:,-1]
    
    
    return train_test_split(X_shuf.astype(None), y_shuf_activity.astype(None), test_size = 0.3, random_state = 0)


