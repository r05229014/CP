import numpy as np
import pickle
import sys
import os 
import time
from sklearn.preprocessing import StandardScaler
#from keras import Model
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
#from keras.utils import multi_gpu_model
# own modile
from Preprocessing import load_alldata, Preprocessing_DNN
from config import ModelMGPU


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
    act = sys.argv[1]
    
    dirx = '../feature/'
    diry = '../target/'
    #X, y = load_data(dirx, diry, case)
    X_train, X_test, y_train, y_test = load_alldata(dirx, diry)
    X_train, X_test, y_train, y_test = Preprocessing_DNN(X_train, X_test, y_train, y_test)
    
    
    if act == 'train':
        # model
        model = DNN()
        parallel_model = ModelMGPU(model, 3)
        parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
        print(model.summary())

        # model save path
        dirpath = "../model/DNN_10layer_512/"
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        
        # path and callbacks
        filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                                    save_best_only=False, period=2)
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        
        # training
        history = parallel_model.fit(X_train , y_train, validation_split=0.1, batch_size=4096, epochs=150, shuffle=True, callbacks = [checkpoint, earlystopper])
        
        # save history<
        history_path = '../history/DNN_10layer_512/'
        if not os.path.exists(history_path):
            os.mkdir(history_path)
        with open(history_path + 'DNN.pkl', 'wb') as f:
            pickle.dump(history.history, f)

    elif act == 'test':
        model_path = sys.argv[2]
        model = load_model(model_path)
        y_pre = model.predict(X_test, batch_size=1024)
        y_pre = y_pre.reshape(137,34,33,33)
        np.save('../predict/DNN/testing.npy', y_pre)
    
    else:
        print('Please type the action you want...')

    
    tEnd = time.time()
    print("It cost %f sec" %(tEnd - tStart))
