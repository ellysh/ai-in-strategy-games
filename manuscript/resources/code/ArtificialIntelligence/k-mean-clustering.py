#!/usr/bin/env python3

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=50,
                       cluster_std=2.00, random_state=0,
                       center_box=(4, 50))

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
kmeans.labels_

plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_.astype(float))

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.xlabel("Просмотры категории автозапчасти")
plt.ylabel("Просмотры категории велозапчасти")
plt.show()

