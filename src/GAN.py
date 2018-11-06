import numpy as np
import sys
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import LeakyReLU, TimeDistributed, LSTM, Input, multiply
from keras.layers.core import Dense, Flatten, Dropout
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import os 
import time
#from keras.utils import multi_gpu_model
# own modile
from Preprocessing import load_alldata, Preprocessing_RNN_vir
from config import ModelMGPU


class CGAN():
    def __init__(self):
        # Input shape
        self.seq_len = 34
        
        optimizer = Adam

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.deiscriminator.compile(loss=['binary_crossentropy'],
                optimizer=optimizer,
                metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        
        # The generator takes noise and the target label as input
        # and generater the corresponding seq of the input
        noise = Input(shape=(self.seq_len))
        label = Input(shape=(1,))
        seq = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity 
        # and the label of that seq
        valid = self.discriminator([img, label])

        # The combined model (stacked generator and discriminator)
        # Trains generator to fool discriminator 
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                optimizer=optimizer)
    
    
    def build_generator(self):

        model = Sequential()

        model.add(LSTM(256, return_sequences=True, input_shape=(self.seq_len,5)))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(TimeDistributed(Dense(1)))

        model.summary()

        noise = Input(shape=(self.seq_len))
        label = Input(shape(1,),dtype='int32')

        model_input = multiply([noise, label])
        img = model(model_input)

        return Model([noise, label], img)

    
    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(256, activation='elu', input_shape=(self.seq_len,)))
        model.add(Dense(256, activation='elu'))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        seq = Input(shape=(self.seq_len,))
        label = Input(shape=(1,), dtype='int32')

        model_input = multiply([seq, label])

        validity = model(model_input)

        return Model([img, label], validity)


    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset

        act = sys.argv[1]
        # path
        dir_x = '../feature/'
        dir_y = '../target/'
        X_train, X_test, y_train, y_test = load_alldata(dir_x, dir_y)
        X_train, X_test, y_train, y_test = Preprocessing_RNN_vir(X_train, X_test, y_train, y_test)

        print(X_train.shape)
        print(y_train.shape)

if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=32, sample_interval=200)
#    tStart = time.time()
#    act = sys.argv[1]
#    # path
#    dir_x = '../feature/'
#    dir_y = '../target/'
#    X_train, X_test, y_train, y_test = load_alldata(dir_x, dir_y)
#    X_train, X_test, y_train, y_test = Preprocessing_RNN_vir(X_train, X_test, y_train, y_test)
#
#    if act == 'train': 
#        model = RNN()
#        parallel_model = ModelMGPU(model, 3)
#        parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
#        print(model.summary())
#        dirpath = "../model/GAN_3_256/"
#        if not os.path.exists(dirpath):
#            os.mkdir(dirpath)
#
#        filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
#        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
#                                     save_best_only=False, period=2)
#        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
#        history = parallel_model.fit(X_train,y_train, validation_split=0.1 , batch_size=256, epochs=150, shuffle=True, callbacks = [checkpoint])
#
#        history_path = '../history/RNN_3_256/'
#        with open(history_path + 'history.pkl', 'wb') as f:
#            pickle.dump(history.history, f)
#
#    elif act == 'test':
#
#        model_path = sys.argv[2]
#        model = load_model(model_path)
#        y_pre = model.predict(X_train, batch_size=1024)
#        y_pre = y_pre.reshape(-1,33,33,34)
#        y_pre = np.swapaxes(y_pre, 1,3)
#        np.save('../predict/RNN/training.npy', y_pre)
#
#    tEnd = time.time()
#
#    print("It cost %f sec" %(tEnd - tStart))
