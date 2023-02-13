import environment
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
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def select_action(state, step_done, policy_net_select):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * step_done / EPS_DECAY)

    step_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            return step_done, torch.argmax(policy_net_select(state)).item()
    else:

        return step_done, torch.tensor(np.random.randint(0, 10), device=device, dtype=torch.long).item()


def select_action2(state, policy_net):
    return torch.argmax(policy_net(state)).item()


def optimize_model(memory, policy_net_agent, optimizer, target_net_agent, n_observation_agent, l1_weight, l2_weight):

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)

    # action_batch = torch.cat(batch.action)

    reward_batch = torch.cat(batch.reward)
    # print(state_batch,action_batch,reward_batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = torch.zeros(BATCH_SIZE, device=device)

    for i in range(BATCH_SIZE):
        state_action_values[i] = policy_net_agent(batch.state[i]).gather(0, batch.action[i])

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = 1
        b = -1
        for i in range(len(next_state_values)):

            if next_state_values[i] != 0:
                b += 1
                if i == 0:
                    b = 0

                a = non_final_next_states[b * n_observation_agent:(b + 1) * n_observation_agent]

                next_state_values[i] = target_net_agent(a).max(0)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    l1_penalty =l1_weight * sum(p.abs().sum() for p in policy_net_agent.parameters())

    l2_penalty =l2_weight * sum(torch.sqrt((p ** 2).sum()) for p in policy_net_agent.parameters())

    loss = criterion(state_action_values,
                     expected_state_action_values) +  l1_penalty +  l2_penalty

    # Optimize the model
    optimizer.zero_grad()

    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net_agent.parameters(), 100)
    optimizer.step()