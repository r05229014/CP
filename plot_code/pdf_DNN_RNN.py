import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import os
import sys
import random

from Preprocessing import *

#np.set_printoptions(threshold=np.nan)

def load_sigma():
    TEST_SPLIT = 0.2
    ls = sorted(os.listdir('./w_p/'))
    
    tags = np.zeros((1,33,33))
    for name in ls:
        if name == 'n01.npy':
            print('Now loading : %s' %name)
            
            tag = np.load('./w_p/' + name)
        
        else:
            print('Now loading : %s' %name)
            tag = np.load('./w_p/' + name)[37:]
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


def test_set_percentile(tags_test):

    t = tags_test
    t = t[t>0]
    a = np.percentile(t,0)
    b = np.percentile(t,20)
    c = np.percentile(t,40)
    d = np.percentile(t,60)
    e = np.percentile(t,80)
    t = tags_test

    a_b = (t>=a) & (t<b) # array where t>=a and t<b
    b_c = (t>=b) & (t<c)
    c_d = (t>=c) & (t<d)
    d_e = (t>=d) & (t<e)
    e_= t>e
    z = t==0

    print('0 :',a)
    print('20 :',b)
    print('40 :',c)
    print('60 :',d)
    print('80 :',e)

    return z, a_b, b_c, c_d, d_e, e_, a,b,c,d,e


tags_train, tags_test = load_sigma()
#dirx = '../feature/'
#diry = '../target/'
#X_train, X_test, y_train, y_DNN = load_alldata(dirx, diry)
#y_DNN = np.load('../predict/DNN/testing.npy')
y_DNN = np.load('../predict/DNN/testing.npy')
print(y_DNN.shape)
p_0, p_0_20, p_20_40, p_40_60, p_60_80, p_80,a,b,c,d,e = test_set_percentile(tags_test)
z = np.loadtxt('../z.txt')

p_0 = p_0.reshape(137,1,33,33)
p_0_20 = p_0_20.reshape(137,1,33,33)
p_20_40 = p_20_40.reshape(137,1,33,33)
p_40_60 = p_40_60.reshape(137,1,33,33)
p_60_80 = p_60_80.reshape(137,1,33,33)
p_80 = p_80.reshape(137,1,33,33)

wqv_p_0 = p_0 * y_DNN*2.5*10**6
wqv_p_0_20 = p_0_20 * y_DNN*2.5*10**6
wqv_p_20_40 = p_20_40 * y_DNN*2.5*10**6
wqv_p_40_60 = p_40_60 * y_DNN*2.5*10**6
wqv_p_60_80 = p_60_80 * y_DNN*2.5*10**6
wqv_p_80 = p_80 * y_DNN*2.5*10**6

mean_wqv_p_0 = np.mean(np.mean(np.mean(wqv_p_0,axis=-1),axis=-1),axis=0)
std_wqv_p_0 = np.std(np.std(np.std(wqv_p_0,axis=-1),axis=-1),axis=0)
std_wqv_p_0 = np.sqrt(std_wqv_p_0**2 * 137*33*33/p_0[p_0==1].shape[0])

mean_wqv_p_0_20 = np.mean(np.mean(np.mean(wqv_p_0_20,axis=-1),axis=-1),axis=0)
std_wqv_p_0_20 = np.std(np.std(np.std(wqv_p_0_20,axis=-1),axis=-1),axis=0)
std_wqv_p_0_20 = np.sqrt(std_wqv_p_0_20**2 * 137*33*33/p_0_20[p_0_20==1].shape[0])

mean_wqv_p_20_40 = np.mean(np.mean(np.mean(wqv_p_20_40,axis=-1),axis=-1),axis=0)
std_wqv_p_20_40 = np.std(np.std(np.std(wqv_p_20_40,axis=-1),axis=-1),axis=0)
std_wqv_p_20_40 = np.sqrt(std_wqv_p_20_40**2 * 137*33*33/p_20_40[p_20_40==1].shape[0])

mean_wqv_p_40_60 = np.mean(np.mean(np.mean(wqv_p_40_60,axis=-1),axis=-1),axis=0)
std_wqv_p_40_60 = np.std(np.std(np.std(wqv_p_40_60,axis=-1),axis=-1),axis=0)
std_wqv_p_40_60 = np.sqrt(std_wqv_p_40_60**2 * 137*33*33/p_40_60[p_40_60==1].shape[0])

mean_wqv_p_60_80 = np.mean(np.mean(np.mean(wqv_p_60_80,axis=-1),axis=-1),axis=0)
std_wqv_p_60_80 = np.std(np.std(np.std(wqv_p_60_80,axis=-1),axis=-1),axis=0)
std_wqv_p_60_80 = np.sqrt(std_wqv_p_60_80**2 * 137*33*33/p_60_80[p_60_80==1].shape[0])

mean_wqv_p_80 = np.mean(np.mean(np.mean(wqv_p_80,axis=-1),axis=-1),axis=0)
std_wqv_p_80 = np.std(np.std(np.std(wqv_p_80,axis=-1),axis=-1),axis=0)
std_wqv_p_80 = np.sqrt(std_wqv_p_80**2 * 137*33*33/p_80[p_80==1].shape[0])


# plot
# 0 percentile
plt.figure(figsize=(6,6))
plt.title('Verticla moisture eddy flux profile when sigma = 0%', fontsize=12)
plt.text(-800,15000,'sigma = 0%')
plt.plot(mean_wqv_p_0, z, linewidth=2.2, label='mean')
plt.fill_betweenx(z,mean_wqv_p_0+std_wqv_p_0, mean_wqv_p_0-std_wqv_p_0, alpha=0.4, label='std')
plt.grid(True)
plt.xlim(-801, 801)
plt.legend(prop={'size': 14})
plt.ylabel('Height[m]', fontsize=12)
plt.xlabel(r"$\overline{w'h'}$[J*m/s*kg]", fontsize=12)
plt.savefig('./percentile/0_DNN.png',dpi=300)
plt.close()

# 0~20 percentile
plt.figure(figsize=(6,6))
plt.title('Verticla moisture eddy flux profile when 0% < sigma < 20%', fontsize=12)
plt.text(-800,15000,'0 percentile sigma = %s'%a+'%')
plt.plot(mean_wqv_p_0_20, z, linewidth=2.2, label='mean')
plt.fill_betweenx(z,mean_wqv_p_0_20+std_wqv_p_0_20, mean_wqv_p_0_20-std_wqv_p_0_20, alpha=0.4, label='std')
plt.grid(True)
plt.xlim(-801, 801)
plt.legend(prop={'size': 14})
plt.ylabel('Height[m]', fontsize=12)
plt.xlabel(r"$\overline{w'h'}$[J*m/s*kg]", fontsize=12)
plt.savefig('./percentile/0~20_DNN.png',dpi=300)
plt.close()


# 20~40 percentile
plt.figure(figsize=(6,6))
plt.title('Verticla moisture eddy flux profile when 20% < sigma < 40%', fontsize=12)
plt.text(-800,15000,'20 percentile sigma = %s'%b +'%')
plt.plot(mean_wqv_p_20_40, z, linewidth=2.2, label='mean')
plt.fill_betweenx(z,mean_wqv_p_20_40+std_wqv_p_20_40, mean_wqv_p_20_40-std_wqv_p_20_40, alpha=0.4, label='std')
plt.grid(True)
plt.xlim(-801, 801)
plt.legend(prop={'size': 14})
plt.ylabel('Height[m]', fontsize=12)
plt.xlabel(r"$\overline{w'h'}$[J*m/s*kg]", fontsize=12)
plt.savefig('./percentile/20~40_DNN.png',dpi=300)
plt.close()


# 40~60 percentile
plt.figure(figsize=(6,6))
plt.title('Verticla moisture eddy flux profile when 40% < sigma < 60%', fontsize=12)
plt.text(-800,15000,'40 percentile sigma = %s'%c+'%')
plt.plot(mean_wqv_p_40_60, z, linewidth=2.2, label='mean')
plt.fill_betweenx(z,mean_wqv_p_40_60+std_wqv_p_40_60, mean_wqv_p_40_60-std_wqv_p_40_60, alpha=0.4, label='std')
plt.grid(True)
plt.xlim(-801, 801)
plt.legend(prop={'size': 14})
plt.ylabel('Height[m]', fontsize=12)
plt.xlabel(r"$\overline{w'h'}$[J*m/s*kg]", fontsize=12)
plt.savefig('./percentile/40~60_DNN.png',dpi=300)
plt.close()


# 60~80 percentile
plt.figure(figsize=(6,6))
plt.title('Verticla moisture eddy flux profile when 60% < sigma < 80%', fontsize=12)
plt.text(-800,15000,'60 percentile sigma = %s'%d+'%')
plt.plot(mean_wqv_p_60_80, z, linewidth=2.2, label='mean')
plt.fill_betweenx(z,mean_wqv_p_60_80+std_wqv_p_60_80, mean_wqv_p_60_80-std_wqv_p_60_80, alpha=0.4, label='std')
plt.grid(True)
plt.xlim(-801, 801)
plt.legend(prop={'size': 14})
plt.ylabel('Height[m]', fontsize=12)
plt.xlabel(r"$\overline{w'h'}$[J*m/s*kg]", fontsize=12)
plt.savefig('./percentile/60~80_DNN.png',dpi=300)
plt.close()


# 80~ percentile
plt.figure(figsize=(6,6))
plt.title('Verticla moisture eddy flux profile when sigma > 80%', fontsize=12)
plt.text(-800,15000,'80 percentile sigma = %s'%e+'%')
plt.plot(mean_wqv_p_80, z, linewidth=2.2, label='mean')
plt.fill_betweenx(z,mean_wqv_p_80+std_wqv_p_80, mean_wqv_p_80-std_wqv_p_80, alpha=0.4, label='std')
plt.grid(True)
plt.xlim(-801, 801)
plt.legend(prop={'size': 14})
plt.ylabel('Height[m]', fontsize=12)
plt.xlabel(r"$\overline{w'h'}$[J*m/s*kg]",fontsize=12)
plt.savefig('./percentile/80~_DNN.png',dpi=300)
plt.close()

# all mean
plt.figure(figsize=(6,6))
plt.title('Verticla moisture eddy flux profile with different sigma', fontsize=12)
plt.plot(mean_wqv_p_0, z, linewidth=2.2, label='sigma = 0%')
plt.plot(mean_wqv_p_0_20, z, linewidth=2.2, label='0% < sigma < 20%')
plt.plot(mean_wqv_p_20_40, z, linewidth=2.2, label='20% < sigma < 40%')
plt.plot(mean_wqv_p_40_60, z, linewidth=2.2, label='40% < sigma < 60%')
plt.plot(mean_wqv_p_60_80, z, linewidth=2.2, label='60% < sigma < 80%')
plt.plot(mean_wqv_p_80, z, linewidth=2.2, label='sigma > 80%')
plt.grid(True)
plt.xlim(0, 150)
plt.legend(prop={'size': 14})
plt.ylabel('Height[m]', fontsize=12)
plt.xlabel(r"$\overline{w'h'}$[J*m/s*kg]",fontsize=12)
plt.savefig('./percentile/all_mean_DNN.png',dpi=300)
plt.close()
