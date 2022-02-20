#!/usr/bin/env python3

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Сгенерировать набор точек
x, y = datasets.make_regression(n_samples=100, n_features=1,
                                random_state=42, noise=15)

# Change direction of the line to descent
x_aligned = [(e + 3) * 800 for e in x]
y_aligned = [23000 - (e + 150) * 23 for e in y]

# Разделить точки на обучающий и тестовый наборы
x_train, x_test, y_train, y_test = train_test_split(x_aligned, y_aligned,
                                                    test_size=0.15)

# Создать модель для линейниной регрессии
regr = linear_model.LinearRegression()

# Обучить модель методом наименьших квадратов
regr.fit(x_train, y_train)

# Получить предсказания модели для тестовго набора данных
y_pred = regr.predict(x_test)

# Подготовить объект Figure
plt.figure(figsize=(8, 6))
plt.xlabel("Пробег, км")
plt.ylabel("Цена,$ ")

# Нарисовать тестовый набор точек
plt.scatter(x_test, y_test, color="black")

# Построить прямую по результатам предсказаний модели
plt.plot(x_test, y_pred, color="blue", linewidth=2)

# Открыть окно с графиком
plt.show()
