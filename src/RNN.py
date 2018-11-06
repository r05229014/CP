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


def RNN():
    print("Build model!!")
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(34,5)))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(LSTM(70, return_sequences=True))
    #model.add(Dense(128, activation='linear'))
    #model.add(Dense(70, activation='linear'))
    return model


if __name__ == '__main__':
    tStart = time.time()
    act = sys.argv[1]
    # path
    dir_x = '../feature/'
    dir_y = '../target/'
    X_train, X_test, y_train, y_test = load_alldata(dir_x, dir_y)
    X_train, X_test, y_train, y_test = Preprocessing_RNN_vir(X_train, X_test, y_train, y_test)

    if act == 'train': 
        model = RNN()
        parallel_model = ModelMGPU(model, 3)
        parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
        print(model.summary())
        dirpath = "../model/GAN_3_256/"
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                                     save_best_only=False, period=2)
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        history = parallel_model.fit(X_train,y_train, validation_split=0.1 , batch_size=256, epochs=150, shuffle=True, callbacks = [checkpoint])

        history_path = '../history/RNN_3_256/'
        with open(history_path + 'history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

    elif act == 'test':

        model_path = sys.argv[2]
        model = load_model(model_path)
        y_pre = model.predict(X_train, batch_size=1024)
        y_pre = y_pre.reshape(-1,33,33,34)
        y_pre = np.swapaxes(y_pre, 1,3)
        np.save('../predict/RNN/training.npy', y_pre)

    tEnd = time.time()

    print("It cost %f sec" %(tEnd - tStart))
