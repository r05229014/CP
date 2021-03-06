import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import os
import sys
import random

from Preprocessing import *
from paper_plot import load_tags

def main():
    dirx = '../feature/'
    diry = '../target/'
    X_train, X_test, y_train, y_test = load_alldata(dirx, diry)
    save_path = '../img/test/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    tags_train, tags_test = load_tags()
    tt = []
    xx = []
    yy = []
    for t in range(tags_test.shape[0]):
        for i in range(tags_test.shape[1]):
            for j in range(tags_test.shape[2]):
                if tags_test[t,i,j] == 0:
                    tt.append(t)
                    xx.append(i)
                    yy.append(j)
    #if plot_tar == 'wqv':
    #    save_path = '../img/virtical/'+ plot_tar +'/'
    #    y_ = y_test
    #if plot_tar == 'Linear':
    #    save_path = '../img/virtical/' + plot_tar + '/'
    y_L = np.load('../predict/LinearRegression/training.npy')
    #elif plot_tar == 'DNN':
    #    save_path = '../img/virtical/' + plot_tar + '/'
    y_D = np.load('../predict/DNN/training.npy')
    y_R = np.load('../predict/RNN/training.npy')
    
    
    z = np.loadtxt('../z.txt')
    #random.seed(777777)
    #tt = [random.randrange(1, 137, 1) for _ in range(10)]
    #xx= [random.randrange(1, 33, 1) for _ in range(10)]
    #yy = [random.randrange(1, 33, 1) for _ in range(10)]

    i = 0
    for t,x,y in zip(tt[0:500],xx[0:500],yy[0:500]):
        plt.figure(i)
        plt.plot(y_train[t,:,x,y]*2.5*10**6, z, label='True')
        plt.plot(y_L[t,:,x,y]*2.5*10**6, z, label='Linear')
        plt.plot(y_D[t,:,x,y]*2.5*10**6, z, label='DNN')
        plt.plot(y_R[t,:,x,y]*2.5*10**6, z, label='RNN')
        
        plt.legend()
        plt.xlim(-301,1001)
        plt.grid(True)
        plt.savefig(save_path + 'cloudy_%s.png' %i)
        plt.close()
        i+=1

if __name__ == '__main__':
    main()
