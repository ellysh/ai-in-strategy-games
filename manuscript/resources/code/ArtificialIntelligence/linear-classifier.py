#!/usr/bin/env python3

from sklearn import datasets
from sklearn.linear_model import SGDClassifier
import copy
import matplotlib.pyplot as plt

# Загрузить набор данных ирисы Фишера
iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target

# Объединить классы ирисов с номерами 1 и 2 в один класс
y_setosa = copy.copy(y)
y_setosa[y_setosa > 0] = 1

# Создать объект классификатора для логистической регрессии
sgdc = SGDClassifier(loss='log', random_state=42)
sgdc.fit(x, y_setosa)

# Вывести ошибку обучения
score = sgdc.score(x, y_setosa)
print("Score: ", score)

# Подготовить объект Figure
plt.figure(figsize=(8, 6))
plt.xlabel("Длина чашелистника, см")
plt.ylabel("Ширина чашелистника, см")

# Нарисовать все точки из обучающего набора
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor="k")

# Нарисовать границу между классами
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
coef = sgdc.coef_
intercept = sgdc.intercept_

def calculate_y(x):
  return (-(x * coef[0, 0]) - intercept[0]) / coef[0, 1]

plt.plot([x_min, x_max], [calculate_y(x_min), calculate_y(x_max)], ls="--", color="blue")

# Открыть окно с графиком
plt.show()