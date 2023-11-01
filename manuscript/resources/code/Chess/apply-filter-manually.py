#!/usr/bin/python3

import numpy as np

sobel_vertical = np.array([ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ])

image = np.array([ [19, 10, 39, 48, 20], [30, 68, 238, 73, 14], [45, 6, 59, 202, 57], [63, 185, 223, 190, 34], [38, 13, 15, 32, 12] ])

tmp = []
row = []
result = []

for i in range(1, 4):
    for j in range(1, 4):
        tmp.append([image[i-1][j-1], image[i-1][j], image[i-1][j+1]])
        tmp.append([image[i][j-1],   image[i][j],   image[i][j+1]])
        tmp.append([image[i+1][j-1], image[i+1][j], image[i+1][j+1]])

        row.append(np.sum(np.multiply(tmp, sobel_vertical)))

        tmp = []

    result.append(row)
    row = []

print(result)

