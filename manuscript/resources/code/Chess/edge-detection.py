#!/usr/bin/python3

import numpy as np
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve


filter1 = np.array([ [1, 1, 1], [0, 0, 0], [-1, -1, -1] ])

img = imread("lena.png")

channels = []
for channel in range(3):
    res = convolve(img[:,:,channel], filter1)
    channels.append(res)

img = np.dstack((channels[0], channels[1], channels[2]))
plt.imshow(img)
plt.show()
