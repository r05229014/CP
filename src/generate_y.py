import pickle
import numpy as np
import skimage.measure
import os

'''
This is a file to generate wqv and wth 
'''

def data_to_y(load_path, name_list, save_path):

    for name in name_list:
        with open (load_path + name, 'rb') as f:
            case = pickle.load(f)
        
        new_dict = {}
        
        for key, value in case.items():
            if key in ['w', 'th', 'qv']:
                print(key, value.shape)
                mean = np.mean(np.mean(value, axis=-1), axis=-1)
                mean2 = np.zeros((73,34,258,258))
                for t in range(73):
                    for z in range(34):
                        mean2[t,z] = np.full((258,258), mean[t,z])
                diff = value - mean2
                #out = np.zeros((73,34,33,33))
                #for t in range(73):
                #    for z in range(34):
                #        #out[t,z] = skimage.measure.block_reduce(diff[t,z], (8,8), np.mean)
                #        out[t,z] = diff[t,z]
                new_dict[key] = diff

        wth = np.zeros((73,34,258,258))
        wqv = np.zeros((73,34,258,258))

        for t in range(73):
            for z in range(34):
                wth[t,z] = new_dict['w'][t,z] * new_dict['th'][t,z]
                wqv[t,z] = new_dict['w'][t,z] * new_dict['qv'][t,z]

        wth_mean = np.zeros((73,34,33,33))
        wqv_mean = np.zeros((73,34,33,33))

        for t in range(73):
            for z in range(34):
                wth_mean[t,z] = skimage.measure.block_reduce(wth[t,z], (8,8), np.mean)
                wqv_mean[t,z] = skimage.measure.block_reduce(wqv[t,z], (8,8), np.mean)

        out_dict = {}
        out_dict['wth'] = wth_mean
        out_dict['wqv'] = wqv_mean
        print('../data/target/' + name)
        with open(save_path + name, 'wb') as fout:
            pickle.dump(out_dict, fout)
path = '../data/'
save_path = '../target/'
name_list = os.listdir(path)
data_to_y(path, name_list, save_path)
