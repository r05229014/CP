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


if __name__ == '__main__':

    tStart = time.time()
    dir_x = '../feature/'
    dir_y = '../target/'
    X_train, X_test, y_train, y_test = load_alldata(dir_x, dir_y)
    X_train, X_test, y_train, y_test = Preprocessing_Linear(X_train, X_test, y_train, y_test)
    #X, y = load_data(dirx, diry, case)

    linear_model = LinearRegression()
    linear_model.fit(X_train,y_train)

    print(linear_model.coef_)
    print(linear_model.intercept_ )

    tEnd = time.time()
    print('It cost %f sec' %(tEnd - tStart))

    # predict
    y_pre = linear_model.predict(X_train)
    y_pre = y_pre.reshape(-1,34,33,33)
    np.save('../predict/LinearRegression/training.npy', y_pre)
