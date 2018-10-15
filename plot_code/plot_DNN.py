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
from Preprocessing import load_alldata, Preprocessing_DNN


def load_data(dirx, diry, case):
    
    with open(dirx + case + '.pkl', 'rb') as x:
        casex = pickle.load(x)
    
    with open(diry + case + '.pkl', 'rb') as y:
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


if __name__ == '__main__':
    # load model adn data
    dirx = '../feature/'
    diry = '../target/'
    model = load_model('../model/DNN_10layer_512/weights-improvement-014-7.222e-09.hdf5')
    save_path = '../predict/DNN_test/'
    
    X_train, X_test, y_train, y_test = load_alldata(dirx, diry)
    X_train, X_test, y_train, y_test = Preprocessing_DNN(X_train, X_test, y_train, y_test)
    
    #print(X_test.shape)
    y_pre = model.predict(X_test, batch_size=1024)
    plt.rcParams['agg.path.chunksize'] = 10000 
    #print(y_pre.shape)
    #y_pre = y_pre.reshape(140,34,33,33)
    plt.scatter(y_test, y_pre)
    plt.title('DNN')
    plt.xlim(-0.01, 0.011)
    plt.ylim(-0.01, 0.01)
    plt.xlabel('True')
    plt.ylabel('Predict')
    plt.savefig(save_path + 'DNN.png')


'''
    all_case = os.listdir('../feature/')
    for case in all_case:
        print('Now processing %s' %case)
        caseName = case[0:3]
        
        # load data
        X_pre,y = load_data(dirx, diry, caseName)
        # pre
        y_pre = model.predict(X_pre, batch_size =1024)
        y_pre = y_pre.reshape(73,34,33,33)
        with open(save_path + caseName + '.pkl', 'wb') as f:
            pickle.dump(y_pre, f)
'''
