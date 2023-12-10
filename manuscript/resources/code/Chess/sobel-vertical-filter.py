#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve

# Объявить двумерный массив 3x3 с вертикальным фильтром Собеля
sobel_vertical = np.array([ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ])

# Загрузить фотографию Лены
img = plt.imread("lena.png")

# Применить фильтр к каждому из 3-х каналов цвета фотографии
channels = []
for channel in range(3):
    result = convolve(img[:,:,channel], sobel_vertical)
    channels.append(result)

# Скомбинировать 3 карты признаков в RGB изображение для визуализации
img = np.dstack((channels[0], channels[1], channels[2]))

# Визуализировать изображение с картами признаков
plt.imshow(img)

# Открыть окно с изображением
plt.show()
