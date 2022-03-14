import abc
import torch

class LearningAgent(abc.ABC):
    @abc.abstractmethod
    def reset(self, env):
        pass

    @abc.abstractmethod
    def get_actions_q(self, state, env):
        pass
    @abc.abstractmethod
    def get_options_q(self, state, action, env):
        pass

    def epsilon_greedy_action(self, env, state, epsilon):
        r = torch.rand((1,)).item()
        if r < epsilon:
            return env.sample()
    
        return self.get_policy(state, env)

    def get_policy(self, state, env):
        action = torch.argmax(self.get_actions_q(state, env)).item()
        t_action = torch.tensor([action], dtype=torch.float)
        option = torch.argmax(self.get_options_q(state, t_action, env)).item()

        return action, option

    @abc.abstractmethod
    def compute_policy(self, env, gamma=0.9, max_iterations=1000000, base_epsilon=0.8, alpha=0.2, debug=False):
        pass
    
    def save_to_file(self, filename):
        pass
    