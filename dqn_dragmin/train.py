import hydra
from util import Logger
from agent.model import LitQNET
from agent.model import LitEnvironment
from agent.policy import EpsilonGreedy
from util import DataVisualization
from tqdm import tqdm
import torch
import os
from constraint import Newton

@hydra.main(config_name='config', config_path='conf')
def main(cfg):
    newton = Newton()
    env = LitEnvironment(cfg.env)
    logger = Logger(cfg)
    model = LitQNET(cfg.model)
    policy = EpsilonGreedy(model)
    data = DataVisualization()
    outputs_dir = os.getcwd() + '/'

    logger.LitQNET_info(model.optimizer, model.loss_function, model.actions_list, model.net.main, model.net.target)
    state, total_reward, init_area, init_objective = env.reset()
    anim_fig = data.anim_reset()

    for epoch in tqdm(range(env.warm_up_epoch), desc='[warm up]'):
        next_state, action_ind, reward, total_reward = env.warm_up(state, model.actions_list, epoch, total_reward, newton, init_area, init_objective)
        data.render_animation(*state)
        model.memory(state, next_state, action_ind, reward)
        state = next_state

    anim_path = outputs_dir + model.warm_up_anim_name
    data.generate_animation(anim_path, model.anim_interval, data.anim_fig)

    for episode in tqdm(range(env.max_episode), desc='[episode]'):
        state, total_reward, init_area, init_objective= env.reset()
        anim_fig = data.anim_reset()
        
        with tqdm(range(env.max_epoch)) as pdar:
            for epoch in pdar:
                pdar.set_description('[train:%d]'%(episode+1))
                action_ind = policy.get_action(state, model.actions_list)
                next_state, action, reward, total_reward = env.step(state, model.actions_list, action_ind, total_reward, newton, init_area, init_objective)
                data.render_animation(*state)
                model.memory(state, next_state, action_ind, reward)
                state = next_state
                
                q_values, td_error, loss = model.train(epoch)

        model.generate_history(total_reward, loss, q_values, td_error)
        if model.train_anim_show:
            anim_path = outputs_dir + 'episode-' + str(episode) + '.GIF'
            data.generate_animation(anim_path, model.anim_interval, data.anim_fig)

    reward_fig_path = outputs_dir + model.reward_fig_name
    loss_fig_path = outputs_dir + model.loss_fig_name
    weight_path = outputs_dir + model.weight_name

    data_frame = data.generate_graph(model.total_rewards_hist, model.losses_hist,  reward_fig_path, loss_fig_path)
    torch.save(model.net.state_dict(), model.weight_name)
    logger.finish_log(data_frame, model.td_errores_hist, model.q_values_hist, weight_path)

if __name__ == '__main__':
    main()
