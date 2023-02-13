from lever import get_cycle
from environment import DQNEnvironment
from Optimizer import DQN
from Optimizer import ReplayMemory
from Optimizer import select_action
from Optimizer import select_action2
from Optimizer import optimize_model
from evaluator import evaluate

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_episodes = 800


data_Noregu = loadtxt('Noregu/policysequence.csv', delimiter=',')
data_L1regu = loadtxt('L1regu/policysequence.csv', delimiter=',')
data_L2regu = loadtxt('L2regu/policysequence.csv', delimiter=',')

partner_cycle = torch.tensor([2, 3, 3, 3, 4, 1], dtype=torch.float32)

N_history = 3
n_dim = 10
agents = 2
state = torch.ones(10)
state = torch.tensor(state, dtype=torch.float32)
state2 = torch.ones(10)
state2 = torch.tensor(state2, dtype=torch.float32)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
partner = False
n_iterations = 10
n_observations = 10 + 10 * N_history * 2
n_actions = 10
step_done = 0
if partner:
  Noregu_ite=[]

  data_Noregu=data_Noregu.reshape(-1)


  n_iterations = 50
  n_observations = 10 + 10 * N_history * 2
  env=DQNEnvironment(state,n_iterations,N_history,partner_cycle)

  env.reset()

  policy_net = DQN(n_observations, n_actions).to(device)
  target_net = DQN(n_observations, n_actions).to(device)
  target_net.load_state_dict(policy_net.state_dict())


  optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
  memory = ReplayMemory(10000)






  reward_train=[]
  reward_test=[]
  policy_levers=[]
  env.partner_cycle = torch.tensor(data_Noregu, dtype=torch.float32)




  for i_episode in range(num_episodes):

      print(i_episode)
      print(policy_levers)


      # Initialize the environment and get it's state
      state= env.reset()
      state = torch.tensor(state, dtype=torch.float32, device=device)
      #cycle=data_Noregu[np.random.randint(5)]
      #env.partner_cycle=torch.tensor(cycle, dtype=torch.float32)



      reward_train.append([])
      reward_test.append([])
      policy_levers=[]

      for t in count():



          step_done,action=select_action(state,step_done,policy_net)
          actions2=select_action2(state,policy_net)
          policy_levers.append(actions2)
          reward_test_value=0

          if actions2==env.partner_cycle[env.episode_step]:
              reward_test_value=1





          observation,reward, terminated= env.step(action)

          reward_train[i_episode].append(reward)

          reward_test[i_episode].append(reward_test_value)


          done = terminated


          reward = torch.tensor([reward], device=device)


          done = terminated
          if terminated:
              next_state = None
          else:
              next_state = torch.tensor(observation, dtype=torch.float32, device=device)



          action=torch.tensor(action, device=device, dtype=torch.long)
          action=action.view(1)

          # Store the transition in memory
          memory.push(state, action, next_state, reward)



          # Move to the next state
          state = next_state

          # Perform one step of the optimization (on the policy network)


          optimize_model(memory, policy_net, optimizer, target_net, n_observations, 0, 0)

          # Soft update of the target network's weights
          # θ′ ← τ θ + (1 −τ )θ′
          target_net_state_dict = target_net.state_dict()
          policy_net_state_dict = policy_net.state_dict()
          for key in policy_net_state_dict:
              target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
          target_net.load_state_dict(target_net_state_dict)

          if done:


              break
  path = 'Noregu/agent_best.h5'
  torch.save(policy_net, path)



all_policy_levers = []
for k in range(10):
    if partner == False:
        env = DQNEnvironment(state, n_iterations, N_history, partner_cycle, partner=False)

        env.reset()

        iterations = n_iterations

        models = []
        memories = []
        policy_nets = []
        target_nets = []
        optimizers = []
        reward_train = []
        reward_test = []
        steps_done = []
        policy_levers = []
        policy_net = 1
        target_net = 1
        optimizer = 1

        policy_levers1 = []

        policy_levers2 = []

        for agent in range(agents):
            policy_netx = DQN(n_observations, n_actions).to(device)
            target_netx = DQN(n_observations, n_actions).to(device)
            target_netx.load_state_dict(policy_netx.state_dict())

            optimizerx = optim.AdamW(policy_netx.parameters(), lr=LR, amsgrad=True)
            memory = ReplayMemory(10000)
            memories.append(memory)
            policy_nets.append(policy_netx)
            target_nets.append(target_netx)
            optimizers.append(optimizerx)

            policy_levers.append([])

        for i_episode in range(num_episodes):
            print('agent:', k, ',', i_episode)
            print(policy_levers1,policy_levers2)

            if i_episode == num_episodes - 2:
                all_policy_levers.append(policy_levers1)

            # print(policy_levers1,policy_levers2)

            policy_levers1 = []
            reward_test.append([])
            policy_levers2 = []
            reward_train.append([])

            optimizer1 = optimizers[1]
            optimizer0 = optimizers[0]
            policy_net0 = policy_nets[0]
            policy_net1 = policy_nets[1]
            target_net0 = target_nets[0]
            target_net1 = target_nets[1]
            memory0 = memories[0]
            memory1 = memories[1]

            # Initialize the environment and get it's state
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            for t in count():

                step_done, action1 = select_action(state, step_done, policy_net0)
                step_done = step_done - 1
                step_done, action2 = select_action(state, step_done, policy_net1)
                action = [action1, action2]

                action1 = torch.tensor(action1, device=device, dtype=torch.long)
                action1 = action1.view(1)

                action2 = torch.tensor(action2, device=device, dtype=torch.long)
                action2 = action2.view(1)

                observation, reward, terminated = env.step(action)

                done = terminated

                reward_train[i_episode].append(reward)

                reward = torch.tensor([reward], device=device, dtype=torch.long)

                done = terminated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device)

                # Store the transition in memory
                memory0.push(state, action1, next_state, reward)
                memory1.push(state, action2, next_state, reward)

                action1 = select_action2(state, policy_net0)
                action2 = select_action2(state, policy_net1)
                if action1 == action2:
                    reward_test[i_episode].append(1)
                else:
                    reward_test[i_episode].append(0)

                policy_levers1.append(action1)
                policy_levers2.append(action2)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)

                optimize_model(memory0, policy_net0, optimizer0, target_net0, n_observations, 0, 0)
                optimize_model(memory1, policy_net1, optimizer1, target_net1, n_observations, 0, 0)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict0 = target_net0.state_dict()
                policy_net_state_dict0 = policy_net0.state_dict()
                target_net_state_dict1 = target_net1.state_dict()
                policy_net_state_dict1 = policy_net1.state_dict()
                for key in policy_net_state_dict0:
                    target_net_state_dict0[key] = policy_net_state_dict0[key] * TAU + target_net_state_dict0[key] * (
                                1 - TAU)
                target_net0.load_state_dict(target_net_state_dict0)
                for key in policy_net_state_dict1:
                    target_net_state_dict1[key] = policy_net_state_dict1[key] * TAU + target_net_state_dict1[key] * (
                                1 - TAU)
                target_net1.load_state_dict(target_net_state_dict1)

                if done:
                    break
        path = 'Noregu/agent{}.h5'.format(k)

        torch.save(policy_net0, path)
all_policy_levers_arr = np.array(all_policy_levers)
savetxt('Noregu/policysequence.csv', all_policy_levers, delimiter=',')

# optimizers[agent]=optimizer
# policy_nets[agent]=policy_net
# target_nets[agent]=target_net
# memories[agent]=memory
# steps_done[agent]=step_done
# _,result_test=get_cross_play(state2,policy_nets,agents,n_iterations,N_history,partner_cycle,partner)
# reward_test.append(result_test)