from .Agent import LearningAgent

import torch

class QMC(LearningAgent):
    def __init__(self, env):
        self.reset(env)
    
    
    def reset(self, env):
        self.actionsQ = {}
        self.optionsQ = {}

    def compute_policy(self, env, gamma=0.9, max_iterations=1000000, base_epsilon=0.8, alpha=0.2, debug=False):
        a_tot_rewards = 0
        o_tot_rewards = 0
        epsilon = base_epsilon
        if debug:
            print("QMC, Training policy...")

        for m in range(max_iterations):
            done = False
            a_tot_reward = 0.0
            o_tot_reward = 0.0

            action_state = env.reset()
            action, option = self.epsilon_greedy_action(env, action_state, epsilon)
            infos = None
            trajectory_op = []
            while not done:
                action_newstate, actions_reward, options_reward, done, infos = env.step(action, option)                
                trajectory_op.append((action_state, action, option, options_reward))
                newaction, newoption = self.epsilon_greedy_action(env, action_newstate, epsilon)

                # Qlearning to learn actions
                self.get_actions_q(action_state, env)[action] += alpha * (actions_reward + gamma * torch.max(self.get_actions_q(action_newstate, env)) - self.get_actions_q(action_state, env)[action])

                action_state, action, option = action_newstate, newaction, newoption

                a_tot_reward += actions_reward
                o_tot_reward += options_reward
            # Monte carlo for to learn options
            G = 0
            for i in range(len(trajectory_op) - 1, -1, -1):
                action_state, action, option, reward = trajectory_op[i]
                G = gamma * G + reward
                self.get_options_q(action_state, action, env)[option] += alpha * (G - self.get_options_q(action_state, action, env)[option])
                
            a_tot_rewards += a_tot_reward
            o_tot_rewards += o_tot_reward



            if debug and ((m+1)%100 == 0):
                a_avg = a_tot_rewards / (m+1)
                o_avg = o_tot_reward / (m + 1)
                print(m+1, a_avg, o_avg, epsilon)
                print(env.algorithm)
                print("------------------------")
                print('actions', env.infos['actions'])
                print("------------------------")
                print('success_rate', env.infos['success_rate'])
                print("------------------------")
                print('algo_results', env.infos['algo_results'])
                print("------------------------")
                print(env.state_infos)
            
            if epsilon > 0:
                epsilon -= base_epsilon/max_iterations
        if debug:
            print("Training finished")
            print("****************************")
            print("success_rate : ", infos['success_rate'])
            print("------------------------")
            print(env.algorithm)
            print("------------------------")
            print('actions', env.infos['actions'])
            print("------------------------")
            print('success_rate', env.infos['success_rate'])
            print("------------------------")
            print('algo_results', env.infos['algo_results'])
            print("------------------------")
            print(env.state_infos)
            print("****************************")
            print(len(self.actionsQ))
            print(len(self.optionsQ))
        
    
    def get_actions_q(self, state, env):
        if state in self.actionsQ:
            return self.actionsQ[state]
        
        temp = torch.rand(env.action_space.n)
        self.actionsQ[state] = temp
        return temp

    def get_options_q(self, state, action, env):
        option_s = (state, action)
        if option_s in self.optionsQ:
            return self.optionsQ[option_s]
        
        temp = torch.rand(env.options_space.n)
        self.optionsQ[option_s] = temp
        return temp

    def get_policy(self, state, env):
        action = torch.argmax(self.get_actions_q(state, env)).item()
        option = torch.argmax(self.get_options_q(state, action, env)).item()

        return action, option