import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import pickle
from Preprocessing import load_alldata, Preprocessing_RNN_vir


if __name__ == '__main__':
    # load model and data
    dirx = '../feature/'
    diry = '../target/'
    model = load_model('../model/DNN_10layer_512/weights-improvement-014-7.222e-09.hdf5')
    save_path = '../predict/DNN_test/'
    
    X_train, X_test, y_train, y_test = load_alldata(dirx, diry)
    X_train, X_test, y_train, y_test = Preprocessing_RNN_vir(X_train, X_test, y_train, y_test)
    model = load_model('../weights-improvement-150-6.982e-09.hdf5')
    y_pre = model.predict(X_test, batch_size=1024)
