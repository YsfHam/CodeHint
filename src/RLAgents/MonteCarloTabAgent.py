from .Agent import LearningAgent

import torch

class MonteCarloTabAgent(LearningAgent):
    def __init__(self, env):
        self.reset(env)

    def reset(self, env):
        self.actionQ = {}
        self.optionQ = {}

    
    def compute_policy(self, env, gamma=0.9, max_iterations=1000000, base_epsilon=0.8, alpha=0.2, debug=False):
        tot_rewards = 0
        epsilon = base_epsilon

        if debug:
            print("Training policy...")
        for m in range(max_iterations):
            done = False
            tot_reward = 0.0

            infos = None
            states_actions = []
            rewards = {}
            reward = 0
            action_state = env.reset()
            while not done:
                t_action_state = torch.tensor(action_state, dtype=torch.float)
                action, option = self.epsilon_greedy_action(env, t_action_state, epsilon)
                newaction_state, reward, done, infos = env.step(action, option)                
                states_actions.append((action_state, action, option))
                rewards[(action_state, action, option)] = reward
                tot_reward += reward
                action_state = newaction_state
            gain = 0
            for i in range(len(states_actions) - 1, -1, -1):
                action_state, action, option = states_actions[i]
                t_action_state = torch.tensor(action_state, dtype=torch.float)
                t_action = torch.tensor([action], dtype=torch.float)
                gain = gamma * gain + rewards[(action_state, action, option)]

                print(self.get_actions_q(t_action_state, env))
                print(self.get_options_q(t_action_state, t_action, env))

                self.actionQ[t_action_state][action] += alpha * (gain - self.get_actions_q(t_action_state, env)[action])
                self.optionQ[(t_action_state, t_action)][option] += alpha * (gain - self.get_options_q(t_action_state, t_action, env)[option])

                
            tot_rewards += tot_reward

            if debug and (((m+1)%100 == 0) or reward == 100):
                avg = tot_rewards / (m+1)
                print(m+1, avg, epsilon)
            
            if epsilon > 0:
                epsilon -= base_epsilon/max_iterations
        if debug:
            print("Training finished")
            print("****************************")
            print("success_rate : ", infos['success_rate'])
            print("------------------------")
            print(env.algorithm)
            print("------------------------")
            print(env.infos)
            print("------------------------")
            print(env.state_infos)
            print("------------------------")
            print(self.actionQ)
            print("------------------------")
            print(self.optionQ)
            print("****************************")


    def get_actions_q(self, state, env):
        if state in self.actionQ:
            return self.actionQ[state]
        self.actionQ[state] = torch.zeros(env.action_space.n)
        return self.actionQ[state]

    def get_options_q(self, state, action, env):
        if (state, action) in self.optionQ:
            return self.optionQ[(state, action)]
        
        self.optionQ[(state, action)] = torch.zeros(env.options_space.n)
        return self.optionQ[(state, action)]
