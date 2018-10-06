import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import os
import sys

def paper_plot(load_path, case, save_path, test):
    print('Now plotting %s' %case)
    with open(load_path + case + '.pkl', 'rb') as f:
        if test == False:
            wqv = pickle.load(f)['wqv']
        else:
            wqv = pickle.load(f)
        
    
    wqv_3000 = wqv[:,12,:,:]
    d = np.load('./w_greater/' + case + '.npy')
    d2 = np.invert(d)
    dd = d.reshape(73,1,33,33)
    dd2 = np.invert(dd)
    
    vir_cloudy = dd * wqv
    hor_cloudy = d * wqv_3000
    vir_nocloud = dd2 * wqv
    hor_nocloud = dd * wqv_3000
    z = np.loadtxt('../z.txt')
    
    x = np.arange(33)
    xx,yy = np.meshgrid(x,x)
    
    for t in range(73):
        plt.figure(t, figsize=(8,10))
        plt.title('t = %s' %t)
        ax1 = plt.subplot2grid((4,2), (0,0), colspan=2, rowspan=2)
        ax1.axis('off')
        cs = ax1.contourf(xx,yy,wqv_3000[t]*2.5*10**6, cmap=cm.Blues, vmax=8000, vmin=0, levels = np.arange(0, 8001, 1000))
        m = plt.cm.ScalarMappable(cmap=cm.Blues)
        m.set_array(wqv_3000)
        m.set_clim(0, 10001)
        cbar = plt.colorbar(cs)
        cbar.set_ticks([0,2000,4000,6000,8000])
        cbar.set_ticklabels([0,2000,4000,6000,8000])
        
        number = np.sum(dd[t])
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
        
        number2 = np.sum(dd2[t])
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
    
        plt.savefig(save_path + case +'/'+'{:0>5d}'.format(t) + '.png')
        plt.close()

if __name__ == '__main__':
    plot_tar = sys.argv[1]

    if plot_tar == 'wqv':
        load_path = '../target/'
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

    all_case = os.listdir(load_path)
    for case in all_case:
        caseName = case[0:3]
        print(caseName)
        dir_name = save_path + caseName
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        paper_plot(load_path, caseName, save_path, test)
    
