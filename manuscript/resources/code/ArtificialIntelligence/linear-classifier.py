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
y_two_classes = copy.copy(y)
y_two_classes[y_two_classes > 0] = 1

# Создать объект классификатора для логистической регрессии
sgdc = linear_model.SGDClassifier(loss='log_loss', random_state=42)
sgdc.fit(x, y_two_classes)

# Вывести ошибку обучения
score = sgdc.score(x, y_two_classes)
print("Score: ", score)

# Подготовить объект Figure
plt.figure(figsize=(8, 6))
plt.xlabel("Длина чашелистика, см")
plt.ylabel("Ширина чашелистика, см")

# Нарисовать все точки из обучающего набора
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor="k")

# Нарисовать границу между классами
x_min, x_max = plt.xlim()
coef = sgdc.coef_
intercept = sgdc.intercept_

def calculate_y(x):
  return (-intercept[0] - (x * coef[0, 0])) / coef[0, 1]

plt.plot([x_min, x_max], [calculate_y(x_min), calculate_y(x_max)], ls="--", color="blue")

# Открыть окно с графиком
plt.show()
