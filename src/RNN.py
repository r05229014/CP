import numpy as np
import sys
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import LeakyReLU, TimeDistributed
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution2D, MaxPooling2D
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.utils import multi_gpu_model
import os 
import time


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


def RNN():
    print("Build model!!")
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(70,5)))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(LSTM(70, return_sequences=True))
    #model.add(Dense(128, activation='linear'))
    #model.add(Dense(70, activation='linear'))
    return model

