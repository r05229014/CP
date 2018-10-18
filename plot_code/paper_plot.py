import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import os
import sys
import random

from Preprocessing import *

def load_tags():
    TEST_SPLIT = 0.2
    ls = sorted(os.listdir('./w_greater/'))
    
    tags = np.zeros((1,33,33))
    for name in ls:
        if name == 'n01.npy':
            print('Now loading : %s' %name)
            
            tag = np.load('./w_greater/' + name)
        
        else:
            print('Now loading : %s' %name)
            tag = np.load('./w_greater/' + name)[37:]
        tags = np.concatenate((tags, tag), axis=0)
    tags = tags[1:]

    # shuffle
    indices = np.arange(tags.shape[0])
    nb_test_samples = int(TEST_SPLIT * tags.shape[0])
    random.seed(777)
    random.shuffle(indices)
    tags = tags[indices]
    tags_train = tags[nb_test_samples:]
    tags_test = tags[0:nb_test_samples]

    print('tags_train shape is : ', tags_train.shape)
    print('tags_test shape is : ', tags_test.shape)
    
    return tags_train, tags_test


def paper_plot(save_path, tags_test, y_test):

    y_test_3000 = y_test[:,12,:,:]

    # tags (boolean array)
    tags_test = tags_test.astype(bool)
    cloudy_vir_tags = tags_test.reshape(137,1,33,33)
    nocloud_vir_tags = np.invert(cloudy_vir_tags)

    vir_cloudy = cloudy_vir_tags * y_test
    vir_nocloud = nocloud_vir_tags * y_test

    # plot
    z = np.loadtxt('../z.txt')
    x = np.arange(33)
    xx,yy = np.meshgrid(x,x)

    for t in range(vir_cloudy.shape[0]):
        print('ploting testing sampling %s' %t)
        plt.figure(t, figsize=(8,10))
        plt.title('testing sample %s' %t)
        ax1 = plt.subplot2grid((4,2), (0,0), colspan=2, rowspan=2)
        ax1.axis('off')
        cs = ax1.contourf(xx,yy,y_test_3000[t]*2.5*10**6, cmap=cm.Blues, vmax=8000, vmin=0, levels = np.arange(0, 8001, 1000))
        m = plt.cm.ScalarMappable(cmap=cm.Blues)
        m.set_array(y_test_3000)
        m.set_clim(0, 10001)
        cbar = plt.colorbar(cs)
        cbar.set_ticks([0,2000,4000,6000,8000])
        cbar.set_ticklabels([0,2000,4000,6000,8000])
        
        number = np.sum(cloudy_vir_tags[t])
        #print(number)
        if number != 0:
            vir = np.sum(np.sum(vir_cloudy[t],axis=-1), axis=-1) / number
        else :
            vir =  np.sum(np.sum(vir_cloudy[t],axis=-1), axis=-1) 
        ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2)
        ax2.set_ylabel('km')
        ax2.set_title('Cloudy grid')
        ax2.plot(vir*2.5*10**6, z/1000)
        ax2.set_xlim(-0.0001*2.5*10**6, 0.003*2.5*10**6)
        ax2.set_ylim(z[0]/1000, z[-1]/1000)
        
        number2 = np.sum(nocloud_vir_tags[t])
        #print(number2)
        if number2 != 0:
            vir2 = np.sum(np.sum(vir_nocloud[t],axis=-1), axis=-1) / number2
        else :
            vir2 =  np.sum(np.sum(vir_nocloud[t],axis=-1), axis=-1) 
        ax3 = plt.subplot2grid((4,2), (2,1), rowspan=2)
        ax3.set_title('No Cloudy grid')
        ax3.plot(vir2*2.5*10**6, z/1000)
        ax3.set_xlim(-0.0001*2.5*10**6, 0.003*2.5*10**6)
        ax3.set_ylim(z[0]/1000, z[-1]/1000)

        plt.savefig(save_path + 'testing' +'/'+'{:0>5d}'.format(t) + '.png')
        plt.close()


def main():
    plot_tar = sys.argv[1]

    if plot_tar == 'wqv':
        save_path = '../img/paper/'+ plot_tar +'/'
        test = False
    elif plot_tar == 'Linear':
        load_path = '../predict/LinearRegression/'
        save_path = '../img/paper/' + plot_tar + '/'
        test = True
    elif plot_tar == 'DNN':
        load_path = '../predict/DNN/'
        save_path = '../img/paper/' + plot_tar + '/'
        test = True
    else:
        sys.exit('Please input wqv, Linear, DNN or RNN you want to plot!!!')


    dir_name = save_path + 'testing/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    paper_plot(save_path, test)


dirx = '../feature/'
diry = '../target/'
X_train, X_test, y_train, y_test = load_alldata(dirx, diry)
tags_train, tags_test = load_tags()
save_path = '../img/paper/'+ 'wqv' +'/'
paper_plot(save_path, tags_test, y_test)
