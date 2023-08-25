import pytorch_lightning as pl
from hydra.utils import instantiate
from torch import nn
import copy
import torch
import numpy as np
from itertools import product
import random
from constraint import Newton
import subprocess
from subprocess import Popen
import csv
from util import DataVisualization
import platform

newton = Newton()
data = DataVisualization()

class LitQNET(pl.LightningModule):
    def __init__(self, cfg_model):
        super().__init__()

        self.qnet = instantiate(cfg_model)
        self.net = Net()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.qnet.learning_rate)
        self.loss_function = torch.nn.SmoothL1Loss()
        self.actions_list = np.array(list(product([-0.01, 0, 0.01], [0])))
        self.epsilon = self.qnet.epsilon
        self.batch_size = self.qnet.batch_size
        self.gamma = self.qnet.gamma
        self.synchro_time = self.qnet.synchro_time
        self.weight_name = self.qnet.weight_name
        self.test_anim_name = self.qnet.test_anim_name
        self.warm_up_anim_name = self.qnet.warm_up_anim_name
        self.anim_interval = self.qnet.anim_interval
        self.reward_fig_name = self.qnet.reward_fig_name
        self.test_reward_fig_name = self.qnet.test_reward_fig_name
        self.loss_fig_name = self.qnet.loss_fig_name
        self.train_anim_show = self.qnet.train_anim_show
        self.buffer, self.total_rewards_hist, self.losses_hist, self.q_values_hist, self.td_errores_hist = [], [], [], [], []

    def memory(self, state, next_state, action_ind, reward):
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action_ind])
        reward = torch.tensor([reward])

        self.buffer.append((state.float(), next_state.float(), action, reward))

    def call_memory(self):
        experience_replay = random.sample(self.buffer, self.batch_size)
        state, next_state, action, reward = map(torch.stack, zip(*experience_replay))

        return state, next_state, action.squeeze(), reward.squeeze()
        
    def calc_current_q_values(self, state, action):
        q_values = self.net(state, nn_model='main')[
            np.arange(self.batch_size), action
        ]

        return q_values

    @torch.no_grad()
    def calc_td_error(self, reward, next_state):
        calculate = self.net(next_state, nn_model='main')
        get_greedy_action = torch.argmax(calculate, axis=1)
        max_q_values = self.net(next_state, nn_model='target')[
            np.arange(self.batch_size), get_greedy_action
        ]
        td_error = (reward + self.gamma * max_q_values).float()

        return td_error

    def update_main_network(self, q_values, td_error):
        loss = self.loss_function(q_values, td_error)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def synchro_target_network(self):
        self.net.target.load_state_dict(self.net.main.state_dict())

    def train(self, epoch):
        if epoch % self.synchro_time == 0 :
            self.synchro_target_network()
        
        state, next_state, action, reward = self.call_memory()
        q_values = self.calc_current_q_values(state, action)
        td_error = self.calc_td_error(reward, next_state)
        loss = self.update_main_network(q_values, td_error)

        return q_values, td_error, loss

    def generate_history(self, total_reward, loss, q_values, td_error):
        print(total_reward)
        self.total_rewards_hist.append(total_reward)
        self.losses_hist.append(loss)
        self.q_values_hist.append(q_values)
        self.td_errores_hist.append(td_error)

        return self.total_rewards_hist

class LitEnvironment(pl.LightningModule):
    def __init__(self, cfg_env):
        super().__init__()
        self.env = instantiate(cfg_env)
        self.init_wide = self.env.init_wide
        self.init_height = self.env.init_height
        self.max_episode = self.env.max_episode
        self.max_epoch = self.env.max_epoch
        self.warm_up_epoch = self.env.warm_up_epoch
        self.round_digit = self.env.round_digit
        self.edp_path = self.env.edp_path
        self.csv_path = self.env.csv_path
        self.init_total_reward = 0
        self.platform = platform.system()
        self.reset()

    def reset(self):
        state = [self.init_wide, self.init_height]
        init_area = np.pi * state[0] * state[1]
        total_reward = self.init_total_reward
        init_objective = self.invoke_FreeFEM(state)

        return state, total_reward, init_area, init_objective

    def calc_area(self, state):
        area = np.pi * state[0] * state[1]
        return area

    def warm_up(self, state, actions_list, epoch, total_reward, newton, init_area, init_objective):
        action_ind = np.random.randint(len(actions_list))
        action = actions_list[action_ind]
        next_state = [round(state[0]+action[0],self.round_digit), round(state[1],self.round_digit)]
        action[1] = newton.newton_method(next_state[0], next_state[1], action[0], init_area)
        next_state = [round(next_state[0],self.round_digit), round(action[1],self.round_digit)]
        objective = self.invoke_FreeFEM(next_state)
        reward = init_objective - objective
        total_reward += reward
        data.debug_log(state, action[0], reward, next_state)

        if epoch > self.max_epoch: state = self.reset()

        return next_state, action_ind, reward, total_reward

    def invoke_FreeFEM(self, state):
        w = state[0]
        h = state[1]
        w_str = str(w)
        h_str = str(h)
        if self.platform=='Windows': 
            subprocess.run(["FreeFEM++", self.edp_path, "-h", h_str, "-w", w_str], encoding="utf-8", stdout=subprocess.PIPE)
        else:
            popen=Popen(["FreeFEM++", self.edp_path, "-h", h_str, "-w", w_str], encoding="utf-8", stdout=subprocess.PIPE)
            popen.wait()
        objective = self.get_objective()

        return objective

    def get_objective(self):
        objective = 0
        with open(self.csv_path, 'r') as f:
            content=csv.reader(f)
            content_dict=[row for row in content]
            for i in range(len(content_dict)):
                if(i>1): objective=-float(content_dict[i][1])

        return objective

    def step(self, state, actions_list, action_ind, total_reward, newton, init_area, init_objective):
        action = actions_list[action_ind]
        next_state = [round(state[0]+action[0],self.round_digit), round(state[1],self.round_digit)]
        action[1] = newton.newton_method(next_state[0], next_state[1], action[0], init_area)
        next_state = [round(next_state[0],self.round_digit), round(action[1],self.round_digit)]
        objective = self.invoke_FreeFEM(next_state)
        reward = init_objective - objective
        total_reward += reward
        data.debug_log(state, action[0], reward, next_state)

        return next_state, action, reward, total_reward

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
        )
        self.target = copy.deepcopy(self.main)

# ************************************************************** #
        for param in self.main.parameters():
            param.requires_grad = True
            print(param)
        print(self.main)

        for param in self.target.parameters():
            param.requires_grad = False
            print(param)
        print(self.target)
# ************************************************************** #

    def forward(self, input, nn_model):
        if nn_model == 'main':
            return self.main(input)
        elif nn_model == 'target':
            return self.target(input)
