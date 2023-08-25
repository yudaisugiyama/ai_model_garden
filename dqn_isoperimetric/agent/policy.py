import numpy as np
import torch

class EpsilonGreedy:
    def __init__(self, model):
        self.model = model
        self.epsilon = self.model.epsilon

    def get_action(self, state, actions_list):
        is_random_action = (np.random.uniform()<self.epsilon)
        if is_random_action:
            q_values = None
            state = torch.tensor(state).unsqueeze(0).float()
            action_ind = np.random.randint(len(actions_list))
        else:
            state = torch.tensor(state).unsqueeze(0).float()
            q_values = self.model.net(state, nn_model='main')
            action_ind = torch.argmax(q_values, axis=1).item()

        return action_ind

class Greedy:
    def __init__(self, model):
        self.model = model
        self.epsilon = self.model.epsilon

    def get_action(self, state, actions_list):
        state = torch.tensor(state).unsqueeze(0).float()
        q_values = self.model.net(state, nn_model='main')
        action_ind = torch.argmax(q_values, axis=1).item()

        return action_ind
