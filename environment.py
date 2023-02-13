from lever import get_cycle
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from numpy import asarray
from numpy import savetxt


from numpy import loadtxt
from typing import List, Optional, Tuple, Union

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class DQNEnvironment:

    def __init__(
            self, state, n_iterations, N_history, partner_cycle, partner=True, include_step=True, include_payoffs=True):
        """Tensor implementation of the iterated lever environment. """
        # Tensor of lever payoffs of shape torch.Size([n_levers])
        self.state = state

        self.partner = partner

        self.partner_cycle = get_cycle(partner_cycle, n_iterations)

        self.n_iterations = n_iterations
        self.episode_length = n_iterations

        self.episode_step = 0

        self.include_step = include_step
        self.include_payoffs = include_payoffs
        self.N_history = N_history

        self.last_player_action = None
        self.last_partner_action = None

        self.state_itself = state

    def reset(self):
        # Reset the environment (including the partner policy).
        self.episode_step = 0
        self.last_player_action = torch.zeros(self.N_history)

        self.last_partner_action = torch.zeros(self.N_history)
        for i in range(self.N_history):
            self.last_player_action[i] = -1
            self.last_partner_action[i] = -1

        state = torch.ones(10)
        state = torch.tensor(state, dtype=torch.float32)
        self.state_itself = state
        a = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        for n in range(self.N_history):
            state = torch.cat((state, a, a), axis=0)

        self.state = state

        return self.state

    def step(self, action):

        if self.partner:
            partner_action = self.partner_cycle[self.episode_step]

            reward = 0.0
            # Compute reward
            if action == partner_action:
                reward = 1.0

            # Update internal environment state
            a = self.N_history - 1
            b = torch.clone(self.last_player_action)
            self.last_player_action[1:self.N_history] = b[:a]
            self.last_player_action[0] = action

            a = self.N_history - 1
            c = torch.clone(self.last_partner_action)
            self.last_partner_action[1:self.N_history] = c[:a]
            self.last_partner_action[0] = partner_action

            self.episode_step += 1
        else:
            action1 = action[0]
            action2 = action[1]
            reward = 0.0
            # Compute reward
            if action1 == action2:
                reward = 1.0

            # Update internal environment state
            a = self.N_history - 1
            b = torch.clone(self.last_player_action)
            self.last_player_action[1:self.N_history] = b[:a]
            self.last_player_action[0] = action1

            a = self.N_history - 1
            c = torch.clone(self.last_partner_action)
            self.last_partner_action[1:self.N_history] = c[:a]
            self.last_partner_action[0] = action2
            self.episode_step += 1

        # Tuple[next_obs: torch.Tensor, reward: float, done: bool]
        return (self._get_obs(), reward, self._is_done())

    def test(self, action):

        partner_action = self.partner_cycle[self.episode_step]

        reward = 0
        # Compute reward
        if action == partner_action:
            reward = 1

        return reward

    def _is_done(self) -> bool:

        return self.episode_step == self.episode_length

    def _get_obs(self) -> torch.Tensor:
        """Return the players observation of the current state. """

        # Only filter None (not zero)
        x = torch.clone(self.state_itself)
        for i in range(self.N_history):
            player_action_flag = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)

            partner_action_flag = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)

            if int(self.last_player_action[i].item()) >= 0:
                player_action_flag[int(self.last_player_action[i].item())] = 1
            x = torch.cat((x, player_action_flag), axis=0)
            if int(self.last_partner_action[i].item()) >= 0:
                partner_action_flag[int(self.last_partner_action[i].item())] = 1
            x = torch.cat((x, partner_action_flag), axis=0)

        # Discriminate between single and multi agent environment setup

        return x

