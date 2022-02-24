#!/usr/bin/env python3

from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import copy

# Загрузить набор данных ирисы Фишера
iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target

# Объединить классы ирисов с номерами 1 и 2 в один класс
y_setosa = copy.copy(y)
y_setosa[y_setosa > 0] = 1

# Создать объект классификатора для логистической регрессии
sgdc = linear_model.SGDClassifier(loss='log', random_state=42)
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
x1_min, x1_max = plt.xlim()
coef = sgdc.coef_
intercept = sgdc.intercept_

def calculate_x2(x1):
  return (-intercept[0] - (x1 * coef[0, 0])) / coef[0, 1]

plt.plot([x1_min, x1_max], [calculate_x2(x1_min), calculate_x2(x1_max)], ls="--", color="blue")

# Открыть окно с графиком
plt.show()
