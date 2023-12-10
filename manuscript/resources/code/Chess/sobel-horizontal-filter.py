#!/usr/bin/python3

import numpy as np
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve

sobel_horizontal = np.array([ [1, 0, -1], [2, 0, -2], [1, 0, -1] ])

img = imread("lena.png")

channels = []
for channel in range(3):
    res = convolve(img[:,:,channel], sobel_horizontal)
    channels.append(res)

img = np.dstack((channels[0], channels[1], channels[2]))
plt.imshow(img, cmap='gray')
plt.show()
