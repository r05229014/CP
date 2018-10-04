import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import pickle

def plot_wqv_vir(case):

    with open('../target/'+ case +'.pkl', 'rb') as f:
        y = pickle.load(f)
    
    z = np.loadtxt('../z.txt')
    wqv = y['wqv']
    count =0
    for t in range(wqv.shape[0]):
        for i in range(wqv.shape[2]):
            for j in range(wqv.shape[3]):

                plt.figure(i)
                plt.title('t = %s' %i)
                plt.plot(wqv[t,:,i,j], z)

                plt.savefig('../img/virtical/wqv/'+ case +'/'+'%08d' %count + '.png')
                plt.close()
                count += 1

case_list = ['n01', 'n02', 'n03', 'n04', 'n05', 'n06', 'n07', 'n08', 'n09',
             's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09']
#for case in case_list:
plot_wqv_vir('n01')
