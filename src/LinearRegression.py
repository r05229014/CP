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


def load_data(dirx, diry, case):
    
    with open(dirx + case, 'rb') as x:
        casex = pickle.load(x)
    
    with open(diry + case, 'rb') as y:
        casey = pickle.load(y)

    # X features concatenate
    for key, value in casex.items():
        casex[key] = value.reshape(73 *34* 33* 33, 1)
    X = np.concatenate((casex['u'], casex['v'], casex['w'], casex['th'], casex['qv']), axis=-1)
    # y target
    y = casey['wqv'].reshape(73*34*33*33, 1)
    
    return X, y

if __name__ == '__main__':

    tStart = time.time()
    dirx = '../feature/'
    diry = '../target/'
    case = 'n01.pkl'
    X, y = load_data(dirx, diry, case)

    linear_model = LinearRegression()
    linear_model.fit(X,y)

    print(linear_model.coef_)
    print(linear_model.intercept_ )

    tEnd = time.time()
    print('It cost %f sec' %(tEnd - tStart))
    
    
