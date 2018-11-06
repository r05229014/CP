import numpy as np
import sys
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LeakyReLU, TimeDistributed
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution2D, MaxPooling2D
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import os 
import time
#from keras.utils import multi_gpu_model
# own modile
from Preprocessing import load_alldata, Preprocessing_RNN_vir
from config import ModelMGPU
import matplotlib.pyplot as plt


def make_data():

    # path
    dir_x = '../feature/'
    dir_y = '../target/'

    X_train, X_test, y_train, y_test = load_alldata(dir_x, dir_y)
    _, _, y_train, y_test = Preprocessing_RNN_vir(X_train, X_test, y_train, y_test)

    L_train = np.load('../predict/LinearRegression/training.npy')
    L_test = np.load('../predict/LinearRegression/testing.npy')
    L_train = np.swapaxes(L_train*2.5*10**6, 1,3).reshape(-1,34)
    L_test = np.swapaxes(L_test*2.5*10**6, 1,3).reshape(-1,34)
    
    R_test = np.load('../predict/RNN/testing.npy')
    R_test = np.swapaxes(R_test*2.5*10**6, 1,3).reshape(-1,34)

    y_train = np.squeeze(y_train*2.5*10**6)
    y_test = np.squeeze(y_test*2.5*10**6)

    X = np.concatenate([y_train,y_test,L_train,L_test,R_test],axis=0)
    y = np.concatenate([np.ones(len(y_train)+ len(y_test), dtype=np.int), np.zeros(len(L_train) + len(L_test) + len(R_test), dtype=np.int)], axis=0)
    
    
    # shuffle
    indices = np.arange(X.shape[0])
    nb_test_samples = int(0.1 * X.shape[0])
    random.seed(666)
    random.shuffle(indices)
    X = X[indices]
    X_train = X[2*nb_test_samples:]
    X_test = X[0:nb_test_samples]
    X_val = X[nb_test_samples:2*nb_test_samples]

    y = y[indices]
    y_train = y[2*nb_test_samples:]
    y_test = y[0:nb_test_samples]
    y_val = y[nb_test_samples:2*nb_test_samples]

    print('X_train shape is : ', X_train.shape)
    print('y_train shape is : ', y_train.shape)
    print('X_test shape is : ', X_test.shape)
    print('y_test shape is : ', y_test.shape)
    print('X_val shape is : ', X_val.shape)
    print('y_val shape is : ', y_val.shape)
    return X_train, X_test, X_val, y_train, y_test, y_val


def classifier():
    print("Build Classifier!!")

    model = Sequential()
    
    model.add(Dense(256, activation = 'relu', kernel_initializer='random_uniform',bias_initializer='zeros', input_shape=(34,)))
    for i in range(2):
        model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
    model.add(Dense(1, activation = 'sigmoid',kernel_initializer='random_uniform',bias_initializer='zeros'))

    return model

if __name__ == '__main__':
    tStart = time.time()
    D_train = np.load('../predict/DNN/training.npy')
    D_train = np.swapaxes(D_train*2.5*10**6,1,3).reshape(-1,34)
    D_y = np.zeros(len(D_train), dtype=np.int)
    # path
    X_train, X_test, X_val, y_train, y_test, y_val = make_data()
    #model = classifier()
    #parallel_model = ModelMGPU(model,3)
    #parallel_model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['acc'])
    #history = parallel_model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=1024, epochs=2, shuffle=True)
    #parallel_model.save('../model/classifier/classifier_1.h5')
    parallel_model = load_model('../model/classifier/classifier_1.h5')
    parallel_model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['acc'])
    val = parallel_model.evaluate(X_test, y_test, batch_size=512)
    val2 = parallel_model.evaluate(D_train, D_y, batch_size=512)
    #print(y_test[y_test==1])
    print(val)
    print(val2)


    
