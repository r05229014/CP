import numpy as np
import sys
import random
import pickle
import os
from sklearn.preprocessing import StandardScaler


def load_alldata(path_x, path_y):
    TEST_SPLIT = 0.2

    l = os.listdir(path_x)
    l = sorted(l)

    # processing X
    X = np.zeros((1,34,33,33,5))
    y = np.zeros((1,34,33,33))

    for name in l:

        if name in ['n01.pkl']:
            print('Now loading : %s' %name)
            
            # X
            with open(path_x + name, 'rb') as f:
                case_x = pickle.load(f)
            for key, value in case_x.items():
                case_x[key] = value.reshape(73,34,33,33,1)
            temp = np.concatenate((case_x['u'], case_x['v'], case_x['w'], case_x['th'], case_x['qv']), axis=-1)

            # y 
            with open(path_y + name, 'rb') as f:
                case_y = pickle.load(f)
            wqv = case_y['wqv']

        else:
            print('Now loading : %s' %name)
            with open(path_x + name, 'rb') as f:
                case_x = pickle.load(f)
            for key, value in case_x.items():
                case_x[key] = value.reshape(73,34,33,33,1)
            temp = np.concatenate((case_x['u'][37:], case_x['v'][37:], case_x['w'][37:], case_x['th'][37:], case_x['qv'][37:]), axis=-1)
            
            # y 
            with open(path_y + name, 'rb') as f:
                case_y = pickle.load(f)
            wqv = case_y['wqv'][37:]

        X = np.concatenate((X, temp), axis=0)
        y = np.concatenate((y, wqv), axis=0)
    X = X[1:]
    y = y[1:]

    # shuffle
    indices = np.arange(X.shape[0])
    nb_test_samples = int(TEST_SPLIT * X.shape[0])
    random.seed(777)
    random.shuffle(indices)
    X = X[indices]
    X_train = X[nb_test_samples:]
    X_test = X[0:nb_test_samples]
    y = y[indices]
    y_train = y[nb_test_samples:]
    y_test = y[0:nb_test_samples]

    print('X_train shape is : ', X_train.shape)
    print('y_train shape is : ', y_train.shape)
    print('X_test shape is : ', X_test.shape)
    print('y_test shape is : ', y_test.shape)

    return X_train, X_test, y_train, y_test


#dirx = '../feature/'
#diry = '../target/'
#load_alldata(dirx, diry)

def Preprocessing_Linear(X_train, X_test, y_train, y_test):
    X_train = X_train.reshape(-1,5)
    X_test = X_test.reshape(-1,5)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    return X_train, X_test, y_train, y_test 


def Preprocessing_DNN(X_train, X_test, y_train, y_test):
    X_train = X_train.reshape(-1,5)
    X_test = X_test.reshape(-1,5)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    sc = StandardScaler()

    # normalize
    for feature in range(5):
        X_train[:,feature:feature+1] = sc.fit_transform(X_train[:, feature:feature+1])
        X_test[:,feature:feature+1] = sc.fit_transform(X_test[:, feature:feature+1])
    
    return X_train, X_test, y_train, y_test 


def Preprocessing_RNN_vir(X_train, X_test, y_train, y_test):

    X_train = X_train.reshape(-1,5)
    X_test = X_test.reshape(-1,5)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    sc = StandardScaler()

    # normalize
    for feature in range(5):
        X_train[:,feature:feature+1] = sc.fit_transform(X_train[:, feature:feature+1])
        X_test[:,feature:feature+1] = sc.fit_transform(X_test[:, feature:feature+1])

    X_train = X_train.reshape(-1,34,33,33,5)
    X_test = X_test.reshape(-1,34,33,33,5)
    y_train = y_train.reshape(-1,34,33,33,1)
    y_test = y_test.reshape(-1,34,33,33,1)

    X_train = np.swapaxes(X_train, 1,3)
    X_test = np.swapaxes(X_test, 1,3)
    y_train = np.swapaxes(y_train, 1,3)
    y_test = np.swapaxes(y_test, 1,3)
    
    X_train = X_train.reshape(-1,34,5)
    X_test = X_test.reshape(-1,34,5)
    y_train = y_train.reshape(-1,34,1)
    y_test = y_test.reshape(-1,34,1)

    print('\nThis is for RNN\'s input! If we assume there are some relationship in vertical!')
    print('X_train shape is : ', X_train.shape)
    print('X_test shape is : ', X_test.shape)
    print('y_train shape is : ', y_train.shape)
    print('y_test shape is : ', y_test.shape)
    return X_train, X_test, y_train, y_test 


