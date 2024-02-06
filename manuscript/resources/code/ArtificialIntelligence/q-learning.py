#!/usr/bin/env python3

import numpy as np
import random

# Define the Tic-Tac-Toe environment
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))  # 3x3 grid representing the Tic-Tac-Toe board
        self.current_player = 1  # 1 for X, -1 for O

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1

    def get_state(self):
        return tuple(self.board.flatten())

    def is_valid_move(self, row, col):
        return self.board[row, col] == 0

    def make_move(self, row, col):
        self.board[row, col] = self.current_player
        self.current_player *= -1  # Switch player

    def is_game_over(self):
        # Check for a win or a draw
        for i in range(3):
            if np.all(self.board[i, :] == 1) or np.all(self.board[:, i] == 1):
                return True, 1  # Player X wins
            elif np.all(self.board[i, :] == -1) or np.all(self.board[:, i] == -1):
                return True, -1  # Player O wins

        if np.all(np.diag(self.board) == 1) or np.all(np.diag(self.board[::-1, :]) == 1):
            return True, 1  # Player X wins

        if np.all(np.diag(self.board) == -1) or np.all(np.diag(self.board[::-1, :]) == -1):
            return True, -1  # Player O wins

        if np.all(self.board != 0):
            return True, 0  # Draw

        return False, None

    def print_board(self):
        for row in self.board:
            print(" ".join(["X" if cell == 1 else "O" if cell == -1 else "-" for cell in row]))
        print()

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, learning_rate=0.2, discount_factor=0.9, exploration_prob=0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = {}  # Q-table to store action-values

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_prob or state not in self.q_table:
            # Explore: choose a random valid action
            valid_actions = [i for i, value in enumerate(state) if value == 0]
            return random.choice(valid_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {i: 0 for i in range(9)}  # Initialize Q-values for the state

        if next_state not in self.q_table:
            self.q_table[next_state] = {i: 0 for i in range(9)}  # Initialize Q-values for the next state

        # Q-learning update rule
        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + \
            self.learning_rate * (reward + self.discount_factor * max(self.q_table[next_state].values()))

# Train the agent to play Tic-Tac-Toe
def train_q_learning_agent(agent1, agent2):
    env = TicTacToe()

    num_episodes = 10000

    move_history = []

    for episode in range(num_episodes):
        move_history.clear()
        env.reset()
        state = env.get_state()
        counter = 0

        if episode % 100 == 0: print(episode)

        while True:
            if counter % 2 == 0:
                action = agent1.choose_action(state)
            else:
                action = agent2.choose_action(state)

            row, col = divmod(action, 3)

            if env.is_valid_move(row, col):
                env.make_move(row, col)
            else:
                # Choose a different action if the selected move is invalid
                continue

            next_state = env.get_state()

            move_history.append((state, action, next_state))

            game_over, winner = env.is_game_over()

            if game_over:
                if winner == 0:
                    reward = 0  # Draw
                else:
                    reward = 1 if winner == 1 else -1  # Player X or O wins

            else:
                reward = 0  # Game still ongoing

            if game_over:
                break

            state = next_state
            counter += 1

        #print("reward = {}".format(reward))
        #print(move_history)

        for move in move_history:
            state = move[0]
            #print(state)
            action = move[1]
            next_state = move[2]
            #print(next_state)
            if reward == 1:
                agent1.update_q_value(state, action, 1, next_state)
                agent2.update_q_value(state, action, -1, next_state)
            elif reward == -1:
                agent1.update_q_value(state, action, -1, next_state)
                agent2.update_q_value(state, action, 1, next_state)
            else:
                agent1.update_q_value(state, action, 0.2, next_state)
                agent2.update_q_value(state, action, 0.5, next_state)


# Test the trained agent
def test_q_learning_agent():
    env = TicTacToe()
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()

    train_q_learning_agent(agent1, agent2)

    num_test_episodes = 10

    for episode in range(num_test_episodes):
        env.reset()
        state = env.get_state()
        counter = 0

        while True:
            if counter % 2 == 0:
                action = agent1.choose_action(state)
            else:
                action = agent2.choose_action(state)

            row, col = divmod(action, 3)

            if env.is_valid_move(row, col):
                env.make_move(row, col)
            else:
                # Choose a different action if the selected move is invalid
                continue

            env.print_board()

            game_over, winner = env.is_game_over()

            if game_over:
                if winner == 0:
                    print("It's a draw!")
                else:
                    print(f"Player {'X' if winner == 1 else 'O'} wins!")
                break

            state = env.get_state()
            counter += 1


test_q_learning_agent()

