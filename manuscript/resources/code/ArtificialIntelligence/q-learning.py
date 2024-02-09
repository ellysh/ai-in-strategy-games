#!/usr/bin/env python3

import numpy as np
import random

# Класс TicTacToe реализует среду для игры в крестики-нолики
class TicTacToe:

    # Инициализация среды
    def __init__(self):
        # Поле для игры размера 3x3 заполнено нулями
        self.board = np.zeros((3, 3))
        # Индекс игрока, который делает ход: 1 для X, -1 для O
        self.current_player = 1

    # Сброс состояния среды в начальное
    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1

    # Получить текущее состояние среды в виде одномерного массива
    def get_state(self):
        return tuple(self.board.flatten())

    # Проверить, что допустим ход на ряд row и столбец col
    def is_valid_move(self, row, col):
        return self.board[row, col] == 0

    # Текущий игрок делает ход на ряд row и столбец col
    def make_move(self, row, col):
        self.board[row, col] = self.current_player
        # Переключить текущего игрока
        self.current_player *= -1

    # Проверка — закончилась ли игра?
    def is_game_over(self):
        # Проходим по полям с 0-ого по 2-е
        for i in range(3):

            # Если в i-ом ряду или i-ом столбце все 1 — выиграл игрок X
            if np.all(self.board[i, :] == 1) or np.all(self.board[:, i] == 1):
                return True, 1

            # Если в i-ом ряду или i-ом  столбце все -1 — выиграл игрок O
            elif np.all(self.board[i, :] == -1) or np.all(self.board[:, i] == -1):
                return True, -1

        # Если все 1 в прямой или обратной диагонали — выиграл игрок X
        if np.all(np.diag(self.board) == 1) or np.all(np.diag(self.board[::-1, :]) == 1):
            return True, 1

        # Если все -1 в прямой или обратной диагонали — выиграл игрок O
        if np.all(np.diag(self.board) == -1) or np.all(np.diag(self.board[::-1, :]) == -1):
            return True, -1

        # Если все игровые поля заняты — ничья
        if np.all(self.board):
            return True, 0

        # В противном случае игра не закончена
        return False, None

    # Вывести на экран состояние доски
    def print_board(self):
        for row in self.board:
            print(" ".join(["X" if cell == 1 else "O" if cell == -1 else "-" for cell in row]))
        print()

# Класс QLearningAgent реализует агента для игры в крестики-нолики
class QLearningAgent:

    # Инициализация агента
    def __init__(self, learning_rate=0.2, discount_factor=0.9, exploration_prob=0.2):
        # α — параметр скорость обучения (learning rate)
        self.learning_rate = learning_rate

        # γ — параметр коэффициент дисконтирования (discount factor)
        self.discount_factor = discount_factor

        # Соотношение исследования и эксплуатации при выборе действия (exploration probability)
        self.exploration_prob = exploration_prob

        # Таблица Q-значений для пар "состояние-действие"
        self.q_table = {}

    # Выбрать действие
    def choose_action(self, state):
        # Если случайное число от 0 до 1 оказалось меньше соотношения исследования и эксплуатации
        # или если указанного состояния state ещё нет в таблице Q-значений агента
        if random.uniform(0, 1) < self.exploration_prob or state not in self.q_table:

            # Исследование: выбрать случайное допустимое действие
            valid_actions = [i for i, value in enumerate(state) if value == 0]
            return random.choice(valid_actions)
        else:
            # Эксплуатация: выбрать действие с максимальный Q-значением для состояния state
            return max(self.q_table[state], key=self.q_table[state].get)

    # Обновить таблицу Q-значений
    def update_q_value(self, state, action, reward, next_state):

        # Если текущего состояния state ещё нет в таблице Q-значений агента
        if state not in self.q_table:

            # Проинициализировать нулями Q-значения для всех действий в состоянии state
            self.q_table[state] = {i: 0 for i in range(9)}

        # Если следующего состояния next_state ещё нет в таблице Q-значений агента
        if next_state not in self.q_table:

            # Проинициализировать нулями Q-значения для всех действий в состоянии next_state
            self.q_table[next_state] = {i: 0 for i in range(9)}

        # Рассчитать рассчитывает новое Q-значение состояния state и действия action
        # по формуле Q-learning
        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + \
            self.learning_rate * (reward + self.discount_factor * max(self.q_table[next_state].values()))

# Функция запуска обучения агента игре крестики-нолики
def train_q_learning_agent(env, agent1, agent2):

    # Установить число эпизодов обучения
    num_episodes = 10000

    # Список с историей ходов каждой игры
    move_history = []

    # Цикл по всем эпизодам обучения (от 0 до num_episodes-1)
    for episode in range(num_episodes):

        # Очистить список с историей ходов
        move_history.clear()

        # Сбросить состояние среды в начальное
        env.reset()

        # Получить текущее состояние среды после сброса
        state = env.get_state()

        # Установить счётчик ходов в ноль
        counter = 0

        # Для каждого 100-ого эпизода выводим на экран число пройденных эпизодов
        if episode % 100 == 0: print(episode)

        # Бесконечный цикл для одной игры
        while True:

            # Если сейчас чётный ход
            if counter % 2 == 0:
                # Действие за игрока X выбирает agent1
                action = agent1.choose_action(state)
            else:
                # Действие за игрока O выбирает agent2
                action = agent2.choose_action(state)

            # Перевести выбранный ход из формата одномерного массива
            # в пару значений ряд (row) и столбец (col)
            row, col = divmod(action, 3)

            # Проверить, что выбранный ход разрешён
            if env.is_valid_move(row, col):

                # Среда выполняет выбранный ход
                env.make_move(row, col)
            else:
                # Выбран неразрешённый ход, поэтому начинаем новую итерацию цикла while
                continue

            # Прочитать новое состояние среды, после выполнения хода action
            next_state = env.get_state()

            # Сохранить в истории ходов прошлое состояние среды state, выбранное действие action и
            # новое состояние среды next_state
            move_history.append((state, action, next_state))

            # Прочитать состояние игры и её результат из объекта env класса среды TicTacToe
            game_over, result = env.is_game_over()

            # Если игра закончилась
            if game_over:

                # Завершить цикл while текущей игры
                break

            # Поменять текущее состояние среды state на новое next_state
            state = next_state

            # Увеличить счётчик ходов на 1
            counter += 1

        # Установить счётчик ходов в истории на ноль
        history_counter = 0

        # Цикл по всем ходам из истории последней игры
        for move in move_history:

            # Переложить данные из триплета move в переменные
            state = move[0]
            action = move[1]
            next_state = move[2]

            # Если чётный ход в истории
            if history_counter % 2 == 0:
                # Обновляем Q-значение для agent1 (X)
                agent = agent1

                # Если игра закончилась не в ничью, вознаграждение равно результату игры.
                # Иначе вознаграждение agent1 (X) равно 0.4
                reward = result if result != 0 else 0.4
            else:
                # Обновить Q-значение для agent2 (O)
                agent = agent2

                # Если игра закончилась не в ничью, вознаграждение равно инвертированному результату игры.
                # Иначе вознаграждение agent2 (O) равно 0.5
                reward = -result if result != 0 else 0.5

            # Обновить таблицу Q-значений агента agent, который сделал проверяемый сейчас ход move
            agent.update_q_value(state, action, reward, next_state)

            # Увеличить счётчик ходов в истории на 1
            history_counter += 1


# Функция тестирования обученного агента
def test_q_learning_agent(env, agent1, agent2):

    # Установить число эпизодов тестирования
    num_test_episodes = 10

    # Цикл по всем эпизодам тестирования (от 0 до num_test_episodes-1)
    for episode in range(num_test_episodes):

        # Сбросить состояние среды в начальное
        env.reset()

        # Получить текущее состояние среды после сброса
        state = env.get_state()

        # Установить счётчик ходов в ноль
        counter = 0

        # Бесконечный цикл для одной игры
        while True:

            # Если сейчас чётный ход
            if counter % 2 == 0:
                # Действие за игрока X выбирает agent1
                action = agent1.choose_action(state)
            else:
                # Действие за игрока O выбирает agent2
                action = agent2.choose_action(state)

            # Перевести выбранный ход из формата одномерного массива
            # в пару значений ряд (row) и столбец (col)
            row, col = divmod(action, 3)

            # Проверить, что выбранный ход разрешён
            if env.is_valid_move(row, col):

                # Среда выполняет выбранный ход
                env.make_move(row, col)
            else:
                # Выбран неразрешённый ход, поэтому начинаем новую итерацию цикла while
                continue

            # Вывести на экран состояние игрового поля после хода
            env.print_board()

            # Прочитать состояние игры и её результат из объекта env класса среды TicTacToe
            game_over, winner = env.is_game_over()

            # Если игра закончена, вывести её результат
            if game_over:
                if winner == 0:
                    print("Ничья!")
                else:
                    print(f"Игрок {'X' if winner == 1 else 'O'} победил!")

                # Завершить цикл while игры
                break

            # Поменять текущее состояние среды state на новое: после хода action
            state = env.get_state()

            # Увеличить счётчик ходов на 1
            counter += 1


# Создать объект среды класса TicTacToe
env = TicTacToe()

# Создать объект первого агента
agent1 = QLearningAgent()

# Создать объект второго агента
agent2 = QLearningAgent()

# Вызвать функцию обучения агента
train_q_learning_agent(env, agent1, agent2)

# Вызвать функцию тестирования агента
test_q_learning_agent(env, agent1, agent2)
