import environment
import Optimizer
import lever
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
from lever import get_cycle
from environment import DQNEnvironment

from Optimizer import select_action
from Optimizer import select_action2
def evaluate(state2, n_iterations, N_history, partner_cycle, policy0, policy1, partner_env,device):
    env2 = DQNEnvironment(state2, n_iterations, N_history, partner_cycle, partner=partner_env)

    st = env2.reset()
    st = torch.tensor(st, dtype=torch.float32, device=device)
    if env2.partner:

        env2.partner_cycle = get_cycle(env2.partner_cycle, n_iterations)


        reward = 0

        episode_step = 0

        for i in range(n_iterations):
            action = select_action2(st, policy0)
            new_state, reward_single, _ = env2.step(action)
            new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
            st = new_state

            reward=reward_single+reward
        return reward
    else:
        reward = 0
        for i in range(n_iterations):
            reward_single = 0

            action1 = select_action2(st, policy0)


            action2 = select_action2(st, policy1)
            action=[action1, action2]
            new_state, reward_single_i, _ = env2.step(action)
            new_state = torch.tensor(new_state, dtype=torch.float32, device=device)

            st = new_state
            # Compute reward

            if action1 == action2:
                reward_single = 1
            reward=reward_single+reward

        return reward


def get_cross_play(state2, policies, pops, n_iterations, N_history, partner_cycle, partner,device):
    results = np.zeros((pops, pops))
    for i in range(pops):

        for j in range(i):
            results[i, j] = evaluate(state2, n_iterations, N_history, partner_cycle, policies[i], policies[j], partner,device)
    results_matrix = results + results.T
    results = results_matrix.sum() / (pops * pops - pops)

    return results_matrix, results
