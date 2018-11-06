import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import os
import sys
import random

from Preprocessing import *

def main():
    dirx = '../feature/'
    diry = '../target/'
    X_train, X_test, y_train, y_test = load_alldata(dirx, diry)
    save_path = '../img/history/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    y_L = np.load('../predict/LinearRegression/training.npy')
    y_D = np.load('../predict/DNN/training.npy')
    y_R = np.load('../predict/RNN/training.npy')
    

    #z = np.loadtxt('../z.txt')
    #random.seed(777777)
    #tt = [random.randrange(1, 137, 1) for _ in range(10)]
    #xx= [random.randrange(1, 33, 1) for _ in range(10)]
    #yy = [random.randrange(1, 33, 1) for _ in range(10)]
    
    y_train = y_train.reshape(-1,1)*2.5*10**6
    y_L = y_L.reshape(-1,1)*2.5*10**6
    y_D = y_D.reshape(-1,1)*2.5*10**6
    y_R = y_R.reshape(-1,1)*2.5*10**6
    
    RMSE_L = np.sqrt(np.mean((y_L - y_train)**2))
    RMSE_D = np.sqrt(np.mean((y_D - y_train)**2))
    RMSE_R = np.sqrt(np.mean((y_R - y_train)**2))
    
    
    y_mean = np.mean(y_train)
    R2_L = 1 - np.sum((y_L - y_train)**2)/np.sum((y_train-y_mean)**2)
    R2_D = 1 - np.sum((y_D - y_train)**2)/np.sum((y_train-y_mean)**2)
    R2_R = 1 - np.sum((y_R - y_train)**2)/np.sum((y_train-y_mean)**2)

    print("RMSE in training set with Linear Regressing: ", RMSE_L)
    print("RMSE in training set with DNN: ",RMSE_D)
    print("RMSE in training set with RNN: ",RMSE_R)
    print("R^2 in training set with Linear Regressing: ",R2_L)
    print("R^2 in training set with DNN: ",R2_D)
    print("R^2 in training set with RNN: ",R2_R)
    print("\n")

    # testinging set
    y_test = y_test.reshape(-1,1)*2.5*10**6
    y_L = np.load('../predict/LinearRegression/testing.npy')
    y_D = np.load('../predict/DNN/testing.npy')
    y_R = np.load('../predict/RNN/testing.npy')
    
    y_L = y_L.reshape(-1,1)*2.5*10**6
    y_D = y_D.reshape(-1,1)*2.5*10**6
    y_R = y_R.reshape(-1,1)*2.5*10**6
    
    RMSE_L = np.sqrt(np.mean((y_L - y_test)**2))
    RMSE_D = np.sqrt(np.mean((y_D - y_test)**2))
    RMSE_R = np.sqrt(np.mean((y_R - y_test)**2))
    
    y_mean = np.mean(y_test)
    R2_L = 1 - np.sum((y_L - y_test)**2)/np.sum((y_test-y_mean)**2)
    R2_D = 1 - np.sum((y_D - y_test)**2)/np.sum((y_test-y_mean)**2)
    R2_R = 1 - np.sum((y_R - y_test)**2)/np.sum((y_test-y_mean)**2)

    print("RMSE in testing set with Linear Regressing: ", RMSE_L)
    print("RMSE in testing set with DNN: ",RMSE_D)
    print("RMSE in testing set with RNN: ",RMSE_R)
    print("R^2 in testing set with Linear Regressing: ",R2_L)
    print("R^2 in testing set with DNN: ",R2_D)
    print("R^2 in testing set with RNN: ",R2_R)

if __name__ == '__main__':
    main()
