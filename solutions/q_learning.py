import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Task: Implement a Deep Q-Network (DQN) to play a simple game (e.g., CartPole-v1)
# 1. Create a Q-network
# 2. Implement experience replay
# 3. Train the agent using the DQN algorithm
# 4. Evaluate the trained agent's performance

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # TODO: Define neural network layers

    def forward(self, state):
        # TODO: Implement forward pass

class DQNAgent:
    def __init__(self, state_size, action_size):
        # TODO: Initialize agent parameters, Q-network, target network, etc.

    def get_action(self, state):
        # TODO: Implement epsilon-greedy action selection

    def train(self):
        # TODO: Implement DQN training loop

# TODO: Set up the environment (gym.make('CartPole-v1'))

# TODO: Train the agent

# TODO: Evaluate the trained agent

# Bonus: Implement Double DQN or Dueling DQN for improved performance