#!/usr/bin/env python3

from sklearn import datasets
from sklearn import cluster
import matplotlib.pyplot as plt

# Сгенерировать набор точек
x, y_true = datasets.make_blobs(n_samples=300, centers=50,
                                cluster_std=2.00, random_state=0,
                                center_box=(4, 50))

# Создать объект для кластеризации точек методом k-средних
kmeans = cluster.KMeans(n_clusters=4, random_state=0)
kmeans.fit(x)

# Подготовить объект Figure
plt.figure(figsize=(8, 6))
plt.xlabel("Просмотры категории автозапчасти")
plt.ylabel("Просмотры категории велозапчасти")

# Нарисовать все точки сгенерированного набора
plt.scatter(x[:,0], x[:,1], c=kmeans.labels_.astype(float))

# Нарисовать центры кластеризации
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

# Открыть окно с графиком
plt.show()
