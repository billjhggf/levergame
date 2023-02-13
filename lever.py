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



def lever_set(n_dim, OP):
    if OP:
        a = np.random.permutation([1, 2, 3, 4, 5, 6, 7, 8, 9])
        a = np.insert(a, 0, 0)
        b = np.random.permutation([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.insert(b, 0, 0)
        c = np.concatenate((a, b), axis=0)

        return (c)
    else:
        return (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))


def get_cycle(partner_cycle, n_iterations):
    a = len(partner_cycle)
    x = torch.clone(partner_cycle)

    b = n_iterations
    div = b // a
    mod = b % a

    if div > 1:
        for n in range(div - 1):
            partner_cycle = torch.cat((partner_cycle, x), axis=0)
        y = torch.clone(partner_cycle)

    if mod > 0:
        partner_cycle_part = x[:mod]

        if div > 1:
            partner_cycle = torch.cat((y, partner_cycle_part), axis=0)
        else:
            partner_cycle = partner_cycle_part

    if len(x) == n_iterations:
        partner_cycle = x

    return partner_cycle
