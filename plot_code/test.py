import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import os
import sys

load_path = '../target/'
all_case = os.listdir(load_path)
case = all_case[0][0:3]
d = np.load('./w_greater/' + case + '.npy')
print(d.shape)
