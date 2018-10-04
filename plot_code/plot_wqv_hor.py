import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import pickle

def plot_wqv(case):

    with open('../target/'+ case +'.pkl', 'rb') as f:
        y = pickle.load(f)
    
    z = np.loadtxt('../z.txt')
    wqv = y['wqv']
    hor_3000 = wqv[:,12,:,:]
    
    x = np.arange(33)
    xx,yy = np.meshgrid(x,x)
    
    for i in range(hor_3000.shape[0]):
        plt.figure(i)
        plt.title('t = %s' %i)
        cs = plt.contourf(xx,yy,hor_3000[i]*2.5*10**6, cmap=cm.Blues, vmax=10000, vmin=0, levels = np.arange(0, 10001, 10))
        #cs = plt.contourf(xx,yy,hor_3000[i]*2.5*10**6, cmap=cm.coolwarm)
        m = plt.cm.ScalarMappable(cmap=cm.Blues)
        m.set_array(hor_3000)
        m.set_clim(0, 10001)
        cbar = plt.colorbar(cs)
        cbar.set_ticks([0,2000,4000,6000,8000,10000])
        cbar.set_ticklabels([0,2000,4000,6000,8000,10000])
        plt.savefig('../img/hor/wqv/'+ case +'/'+'{:0>5d}'.format(i) + '.png')
        plt.close()

case_list = ['n01', 'n02', 'n03', 'n04', 'n05', 'n06', 'n07', 'n08', 'n09',
             's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09']
for case in case_list:
    plot_wqv(case)
