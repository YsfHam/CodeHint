import abc
import torch

class LearningAgent(abc.ABC):
    @abc.abstractmethod
    def reset(self, env):
        pass

    @abc.abstractmethod
    def get_q(self, state, env):
        pass

    def epsilon_greedy_action(self, env, state, epsilon):
        r = torch.rand((1,)).item()
        if r < epsilon:
            return env.actions_space.sample()
    
        return self.get_policy(state, env)

    def get_policy(self, state, env):
        return torch.argmax(self.get_q(state, env)).item()

    @abc.abstractmethod
    def compute_policy(self, env, gamma=0.9, max_iterations=1000000, base_epsilon=0.8, alpha=0.2, debug=False):
        pass
    