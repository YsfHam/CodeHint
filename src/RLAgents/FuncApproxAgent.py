import torch
from .Agent import LearningAgent

class FuncApproxAgent(LearningAgent):
    def __init__(self, env):
        self.reset(env)

    def reset(self, env):
        self.actions_parameters = torch.rand((env.action_space.n, env.state_len), dtype=torch.float64)
        self.options_parameters = torch.rand((env.options_space.n, env.state_len + 1), dtype=torch.float64)


    def get_actions_q(self, state, env):
        return self.actions_parameters @ state
    def get_options_q(self, state, action, env):
        return self.options_parameters @ torch.cat((state, action))

    def save_to_file(self, filename):
        torch.save(self.actions_parameters, filename[0])
        torch.save(self.options_parameters, filename[1])