from lever import get_cycle
from environment import DQNEnvironment
from Optimizer import DQN
from Optimizer import ReplayMemory
from Optimizer import select_action
from Optimizer import select_action2
from Optimizer import optimize_model

from evaluator import get_cross_play

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
from mpl_toolkits.axes_grid1 import ImageGrid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


agent=5
loaded_model_Noregu=[]
loaded_model_L1regu=[]
loaded_model_L2regu=[]
for k in range(agent):
    path_Noregu = 'Noregu/agent{}.h5'.format(k)

    path_L1regu = 'L1regu/agent{}.h5'.format(k)
    path_L2regu = 'L2regu/agent{}.h5'.format(k)


    loaded_model_Noregu_single = torch.load(path_Noregu).to(device)


    loaded_model_L1regu_single = torch.load(path_L1regu).to(device)

    loaded_model_L2regu_single = torch.load(path_L2regu).to(device)

    loaded_model_L2regu.append(loaded_model_L2regu_single)
    loaded_model_L1regu.append(loaded_model_L1regu_single)
    loaded_model_Noregu.append(loaded_model_Noregu_single)


loaded_model_L1regu_single = torch.load('L1regu/agent_best.h5').to(device)
loaded_model_L1regu.append(loaded_model_L1regu_single)
loaded_model_L2regu_single = torch.load('L2regu/agent_best.h5').to(device)
loaded_model_L2regu.append(loaded_model_L2regu_single)
loaded_model_Noregu_single = torch.load('Noregu/agent_best.h5').to(device)
loaded_model_Noregu.append(loaded_model_Noregu_single)



partner_cycle = torch.tensor([2, 3, 3, 3, 4, 1], dtype=torch.float32)
n_iterations = 10
n_actions = 10
n_history=3
n_observations = 10 + 10 * n_history * 2
partner=False
state2 = torch.ones(10)
state2 = torch.tensor(state2, dtype=torch.float32, device=device)

loaded_model_L2regu_graph,_=get_cross_play(state2, loaded_model_L2regu, agent+1, n_iterations, n_history, partner_cycle, partner,device)
loaded_model_L1regu_graph,_=get_cross_play(state2, loaded_model_L1regu, agent+1, n_iterations, n_history, partner_cycle, partner,device)
loaded_model_Noregu_graph,_=get_cross_play(state2, loaded_model_Noregu, agent+1, n_iterations, n_history, partner_cycle, partner,device)
cmap = 'hot'
# ax = plt.figure().gca()


# Set up figure and image grid
fig = plt.figure(figsize=(12, 2))


grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,3),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )

# Add data to image grid

grid[0].imshow(loaded_model_Noregu_graph, vmin=0, vmax=+10,cmap = cmap)
grid[0].set_title('a) \n No regularization cross play')
#compressibility:1.25
#compressibility:9.1
grid[1].imshow(loaded_model_L1regu_graph, vmin=0, vmax=+10,cmap = cmap)
grid[1].set_title('b) \n L1 regularization-0.001 cross play')
#compressibility:9.3
ax = grid[2]
im = ax.imshow(loaded_model_L2regu_graph, vmin=0, vmax=+10,cmap = cmap)
grid[2].set_title('g) \n L2 regularization-0.01 cross play')



# Colorbar
ax.cax.colorbar(im)
ax.cax.toggle_label(True)

plt.tight_layout()
plt.show()
