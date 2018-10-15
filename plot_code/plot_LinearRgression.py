import numpy as np
import sys
import random
from sklearn.preprocessing import StandardScaler
import os 
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import time
from Preprocessing import load_alldata, Preprocessing_Linear


def load_data(dirx, diry, case):
    
    with open(dirx + case + '.pkl', 'rb') as x:
        casex = pickle.load(x)
    
    with open(diry + case + '.pkl', 'rb') as y:
        casey = pickle.load(y)

    # X features concatenate
    for key, value in casex.items():
        casex[key] = value.reshape(73 *34* 33* 33, 1)
    X = np.concatenate((casex['u'], casex['v'], casex['w'], casex['th'], casex['qv']), axis=-1)
    # y target
    y = casey['wqv'].reshape(73*34*33*33, 1)
    
    return X, y
'''
if __name__ == '__main__':

    tStart = time.time()
    dirx = '../feature/'
    diry = '../target/'
    case = 'n01'
    X, y = load_data(dirx, diry, case)

    linear_model = LinearRegression()
    linear_model.fit(X,y)

    print(linear_model.coef_)
    print(linear_model.intercept_ )

    tEnd = time.time()
    print('It cost %f sec' %(tEnd - tStart))

    # Plot
    save_path = '../predict/LinearRegression/'

    all_case = os.listdir('../feature/')
    for case in all_case:
        print('Now procession %s' %case)
        caseName = case[0:3]
        
        # load data
        X_pre,y = load_data(dirx, diry, caseName)
        # pre
        y_pre = linear_model.predict(X_pre)
        y_pre = y_pre.reshape(73,34,33,33)
        with open(save_path + caseName + '.pkl', 'wb') as f:
            pickle.dump(y_pre, f)
'''

if __name__ == '__main__':

    tStart = time.time()
    dirx = '../feature/'
    diry = '../target/'
    X_train, X_test, y_train, y_test = load_alldata(dirx, diry)
    X_train, X_test, y_train, y_test = Preprocessing_Linear(X_train, X_test, y_train, y_test)

    linear_model = LinearRegression()
    linear_model.fit(X_train,y_train)

    print(linear_model.coef_)
    print(linear_model.intercept_ )

    tEnd = time.time()
    print('It cost %f sec' %(tEnd - tStart))

    # Plot
    save_path = '../predict/LinearRegression_test/'
    plt.rcParams['agg.path.chunksize'] = 10000 
    y_pre = linear_model.predict(X_test)
    plt.scatter(y_test, y_pre)
    plt.xlim(-0.02, 0.01)
    plt.ylim(-0.02, 0.05)
    plt.title('Linear Regression')
    plt.xlabel('True')
    plt.ylabel('Predict')
    plt.savefig(save_path + 'Linear.png')

