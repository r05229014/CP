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
    save_path = '../img/virtical/test/'

    #if plot_tar == 'wqv':
    #    save_path = '../img/virtical/'+ plot_tar +'/'
    #    y_ = y_test
    #if plot_tar == 'Linear':
    #    save_path = '../img/virtical/' + plot_tar + '/'
    y_L = np.load('../predict/LinearRegression/testing.npy')
    #elif plot_tar == 'DNN':
    #    save_path = '../img/virtical/' + plot_tar + '/'
    y_D = np.load('../predict/DNN/testing.npy')
    y_R = np.load('../predict/RNN/testing.npy')
    

    z = np.loadtxt('../z.txt')
    random.seed(777777)
    tt = [random.randrange(1, 137, 1) for _ in range(10)]
    xx= [random.randrange(1, 33, 1) for _ in range(10)]
    yy = [random.randrange(1, 33, 1) for _ in range(10)]

    for t,x,y in zip(tt,xx,yy):
        plt.figure(t)
        plt.plot(y_test[t,:,x,y]*2.5*10**6, z, label='True')
        plt.plot(y_L[t,:,x,y]*2.5*10**6, z, label='Linear')
        plt.plot(y_D[t,:,x,y]*2.5*10**6, z, label='DNN')
        plt.plot(y_R[t,:,x,y]*2.5*10**6, z, label='RNN')
        
        plt.legend()
        plt.xlim(-300,800)
        plt.savefig(save_path + '%s.png' %t)
        plt.close()

if __name__ == '__main__':
    main()
    #    save_path = '../img/virtical/'+ plot_tar +'/'
