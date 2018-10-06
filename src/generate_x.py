import pickle
import numpy as np
import skimage.measure
import os

'''
This is a file to generate mean X
'''

def data_to_y(load_path, name_list, save_path):

    for name in name_list:
        print('\n Now processing %s ' %name)
        with open (load_path + name, 'rb') as f:
            case = pickle.load(f)

        tmp_dict = {}
        for key, value in case.items():
            print(key, value.shape)
            mean = np.zeros((73,34,33,33))
            
            for t in range(73):
                for z in range(34):
                    mean[t,z] = skimage.measure.block_reduce(value[t,z], (8,8), np.mean)
            
            tmp_dict[key] = mean

        with open(save_path + name, 'wb') as f:
           pickle.dump(tmp_dict, f)


if __name__ == '__main__':
    path = '../data/'
    save_path = '../feature/'
    name_list = os.listdir(path)
    print('The data we want to traslate to feature/ : %s' %name_list)
    data_to_y(path, name_list, save_path)
