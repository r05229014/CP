import numpy as np

d = np.load('./w_greater/n01.npy')

for t in range(73):
    print(d[t][d[t]>0], t)


print(d.shape)
