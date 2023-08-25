from util import Logger
from util import DataVisualization
from agent.model import LitEnvironment
from agent.model import LitQNET
from agent.policy import Greedy
import hydra
from tqdm import tqdm
import pandas as pd
import torch
from pathlib import Path
import os
from constraint import Newton
import csv

@hydra.main(config_name='config', config_path='conf')
def main(cfg):
    newton = Newton()
    env = LitEnvironment(cfg.env)
    logger = Logger(cfg)
    model = LitQNET(cfg.model)
    policy = Greedy(model)
    data = DataVisualization()
    actions_hist, reward_hist, states_hist = [], [], []
    weight_path = Path(input('Input path of weight.pth-->'))
    param = torch.load(weight_path)
    model.net.load_state_dict(param)
    print(param)
    outputs_dir = os.getcwd() + '/'

    state, total_reward , init_area, init_objective= env.reset()
    data.anim_reset()

    with tqdm(range(env.max_epoch)) as pdar:
            for epoch in pdar:
                pdar.set_description('[test]')
                action_ind = policy.get_action(state, model.actions_list)
                next_state, action, reward, total_reward = env.step(state, model.actions_list,action_ind, total_reward, newton, init_area, init_objective)

                data.output_csv(epoch, state, action, reward, next_state)
                data.render_animation(*state)
                actions_hist.append(action)
                reward_hist.append(reward)
                states_hist.append(state)

                state = next_state

    print(total_reward)

    anim_path = outputs_dir + model.test_anim_name
    test_reward_fig_path = outputs_dir + model.test_reward_fig_name

    data.generate_animation(anim_path, model.anim_interval, data.anim_fig)
    data_frame = data.generate_test_data_frame(actions_hist, reward_hist, states_hist, test_reward_fig_path)
    logger.finish_log(data_frame, model.td_errores_hist, model.q_values_hist, weight_path)

    data.result_fig(data_frame)

if __name__ == '__main__':
    main()