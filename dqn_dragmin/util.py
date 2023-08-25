from logging import getLogger
from matplotlib import animation
from omegaconf import OmegaConf
import time
from decimal import Decimal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from csv import writer

class Logger():
    def __init__(self, cfg):
        self.logger = getLogger(__name__)
        self.logger.info('yaml\n%s', OmegaConf.to_yaml(cfg))
        self.start_time = time.time()

    def LitQNET_info(self, optimizer, loss_function ,actions_list, main_network, target_network):
        self.logger.info('optimizer\n%s', optimizer)
        self.logger.info('loss function\n%s', loss_function)
        self.logger.info('actions list\n%s', actions_list)
        self.logger.info('main network\n%s', main_network)
        self.logger.info('target network\n%s', target_network)

    def finish_log(self, data_frame, td_errores_hist, q_values_hist, path):
        self.logger.info('q value\n%s', q_values_hist)
        self.logger.info('td error\n%s', td_errores_hist)
        self.logger.info('total reward history\n%s', data_frame)
        self.logger.info('weight path\n%s', path)
        processing_time = str(Decimal(time.time()).quantize(Decimal('0.001'), rounding='ROUND_HALF_UP') - Decimal(self.start_time).quantize(Decimal('0.001'), rounding='ROUND_HALF_UP'))
        self.logger.info('time\n%ss', processing_time)

class DataVisualization():
    def __init__(self):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['ytick.major.width'] = 1.0
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.linewidth'] = 1.0
        # sns.set(style='whitegrid')

    def reward_fig(self, data, path):
        self.anim_reset()
        plt.xlabel('episode[times]')
        plt.ylabel('total reward[-]')

        reward_ax = sns.lineplot(data=data)
        reward_ax.legend(loc='lower left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(path)

    def test_reward_fig(self, data, path):
        self.anim_reset()
        plt.xlabel('epoch[times]')
        plt.ylabel('reward[-]')

        reward_ax = sns.lineplot(data=data)
        reward_ax.legend(loc='lower left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(path)

    def loss_fig(self, data, path):
        self.anim_reset()
        plt.xlabel('episode[times]')
        plt.ylabel('loss[-]')

        loss_ax = sns.lineplot(data=data)
        loss_ax.legend(loc='lower left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(path)

    def generate_graph(self, total_rewards_hist, losses_hist, reward_fig_path, loss_fig_path):
        data_frame = pd.DataFrame({
            'total reward' : total_rewards_hist,
            'loss' : losses_hist
        })

        reward_data = data_frame['total reward']
        loss_data = data_frame['loss']

        self.reward_fig(reward_data, reward_fig_path)
        self.loss_fig(loss_data, loss_fig_path)

        return data_frame

    def generate_test_data_frame(self, actions_hist, reward_hist, states_hist, path):
        data_frame = pd.DataFrame({
            'state' : states_hist,
            'action' : actions_hist,
            'reward' : reward_hist
        })

        reward_data = data_frame['reward']
        self.test_reward_fig(reward_data, path)

        return data_frame

    def anim_reset(self):
        self.anim_fig, self.anim_ax = plt.subplots()
        self.imgs_hist = []

        return self.anim_fig, self.anim_ax

    def draw_ellipse(self, a, b):
        theta = np.linspace(0, 2*np.pi, 100)
        x = a * np.cos(theta)
        y = b * np.sin(theta)

        return x, y

    def render_animation(self, a, b):
        x, y = self.draw_ellipse(a, b)
        self.anim_ax.set_aspect('equal')
        img = self.anim_ax.plot(x, y, color='blue')
        self.imgs_hist.append(img)

        return img

    def generate_animation(self, anim_path, interval, fig):
        anim = animation.ArtistAnimation(fig, self.imgs_hist, interval=interval)
        anim.save(anim_path)

    def result_fig(self, data_frame):
        idx = data_frame['reward'].idxmax()
        print(data_frame.iloc[idx,0])
        x = data_frame.iloc[idx,0][0]
        y = data_frame.iloc[idx,0][1]
        self.anim_reset()
        img = self.render_animation(x, y)
        plt.savefig('result.png')

    def debug_log(self, state, action, reward, next_state):
        print('+----------------------------+--------+-----------------------+---------------------------+')
        print('| state                      | action | reward                | next state                |')
        print('|',state,'| ',action,' | ',reward,' |',next_state,'|')

    def output_csv(self, epoch, state, action, reward, next_state):
        data_list = [epoch, state[0], state[1], action[0], reward, next_state[0], next_state[1]]
        with open('log.csv','a', newline='', encoding='utf-8') as f:
            write_data = writer(f)
            write_data.writerow(data_list)
            f.close()