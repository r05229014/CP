import numpy as np
import sys
import random
import pickle
import os


def load_alldata(path_x, path_y):
    TEST_SPLIT = 0.2

    l = os.listdir(path_x)
    l = sorted(l)
    print(path_x + l[0])

    # processing X
    X_train = np.zeros((1,34,33,33,5))
    X_test = np.zeros((1,34,33,33,5))
    y_train = np.zeros((1,34,33,33))
    y_test = np.zeros((1,34,33,33))

    for name in l:

        if name in ['n01.pkl', 's01.pkl']:
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
            
            # shuffle
            indices = np.arange(temp.shape[0])  
            nb_test_samples = int(TEST_SPLIT * temp.shape[0])
            np.random.shuffle(indices)
            temp = temp[indices]
            train = temp[nb_test_samples:]
            test = temp[0:nb_test_samples]

            wqv = wqv[indices]
            train_wqv = wqv[nb_test_samples:]
            test_wqv = wqv[0:nb_test_samples]


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

            # shuffle
            indices = np.arange(temp.shape[0])  
            nb_test_samples = int(TEST_SPLIT * temp.shape[0])
            np.random.shuffle(indices)
            temp = temp[indices]
            train = temp[nb_test_samples:]
            test = temp[0:nb_test_samples]

            wqv = wqv[indices]
            train_wqv = wqv[nb_test_samples:]
            test_wqv = wqv[0:nb_test_samples]

        X_train = np.concatenate((X_train, train), axis=0)
        X_test = np.concatenate((X_test, test), axis=0)
        y_train = np.concatenate((y_train, train_wqv), axis=0)
        y_test = np.concatenate((y_test, test_wqv), axis=0)

    X_train = X_train[1:]
    X_test = X_test[1:]
    y_train = y_train[1:]
    y_test = y_test[1:]
    print('X_train shape is : ', X_train.shape)
    print('X_test shape is : ', X_test.shape)
    print('y_train shape is : ', y_train.shape)
    print('y_test shape is : ', y_test.shape)
def Preprocessing_RNN():
    pass

path_x = '../feature/'
path_y = '../target/'
load_alldata(path_x, path_y)
