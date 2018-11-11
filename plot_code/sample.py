import numpy as np
import pickle
import skimage.measure
import os

np.set_printoptions(threshold=np.nan)
def sample_w(path, case):
    with open(path + case + '.pkl', 'rb') as f:
        d = pickle.load(f)
    w = d['w']
    w_3000 = w[:,12,:,:] > 0.5
    w_ = np.zeros((73,33,33))
    for t in range(w_3000.shape[0]):
        w_[t] = skimage.measure.block_reduce(w_3000[t,], (8,8), np.sum)
        w_[t] = w_[t]*100/64
    #out = w_>16
    np.save('./w_p/' + case, w_)


if __name__ == '__main__':
    path = '../data/'
    name_list = os.listdir(path)
    for name in name_list:
        case = name[0:3]
        sample_w(path, case)
#    sample_w(path, case)
