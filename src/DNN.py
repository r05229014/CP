import numpy as np
import pickle
import sys
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import Model
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.utils import multi_gpu_model
import os 
import time


def load_data(dirx, diry, case):
    
    with open(dirx + case, 'rb') as x:
        casex = pickle.load(x)
    
    with open(diry + case, 'rb') as y:
        casey = pickle.load(y)

    # X features concatenate

    sc = StandardScaler()
    for key, value in casex.items():
        casex[key] = value.reshape(73*34*33*33, 1)
        casex[key] = sc.fit_transform(casex[key])
    X = np.concatenate((casex['u'], casex['v'], casex['w'], casex['th'], casex['qv']), axis=-1)

    # y target
    y = casey['wqv'].reshape(73*34*33*33, 1)
    
    return X, y

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)
            
        return super(ModelMGPU, self).__getattribute__(attrname)

def DNN():
    print("Build model!!")
    model = Sequential()
    
    model.add(Dense(256, activation = 'relu', kernel_initializer='random_uniform',bias_initializer='zeros', input_shape=(5,)))
    for i in range(10):
        model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
        #model.add(LeakyReLU(alpha=0.1))
        #model.add(BatchNormalization())
    model.add(Dense(1, activation = 'linear',kernel_initializer='random_uniform',bias_initializer='zeros'))
    return model


if __name__ == '__main__':
    tStart = time.time()

    dirx = '../feature/'
    diry = '../target/'
    case = 'n01.pkl'
    X, y = load_data(dirx, diry, case)
    model = DNN()
    parallel_model = ModelMGPU(model, 3)
    parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
    print(model.summary())
    dirpath = "../model/DNN/"
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    
    filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                                save_best_only=False, period=5)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    parallel_model.fit(X,y, validation_split=0.1, batch_size=4096, epochs=150, shuffle=True, callbacks = [checkpoint, earlystopper])
    
    tEnd = time.time()
    
    print("It cost %f sec" %(tEnd - tStart))
