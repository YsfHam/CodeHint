from .Agent import LearningAgent

import torch

class QLearningAgentTab(LearningAgent):
    def __init__(self, env):
        self.reset(env)
    
    
    def reset(self, env):
        self.actionsQ = {}
        self.optionsQ = {}

    def compute_policy(self, env, gamma=0.9, max_iterations=1000000, base_epsilon=0.8, alpha=0.2, debug=False):
        tot_rewards = 0
        epsilon = base_epsilon
        winning = 0
        if debug:
            print("Training policy...")

        for m in range(max_iterations):
            done = False
            tot_reward = 0.0

            action_state = env.reset()
            t_action_state = torch.tensor(action_state, dtype=torch.float64)
            action, option = self.epsilon_greedy_action(env, t_action_state, epsilon)
            infos = None
            reward = 0
            while not done:
                t_action = torch.tensor([action], dtype=torch.float64)
                action_newstate, actions_reward, options_reward, done, infos = env.step(action, option)                
                t_action_newstate = torch.tensor(action_newstate, dtype=torch.float64)
                newaction, newoption = self.epsilon_greedy_action(env, t_action_newstate, epsilon)
                t_newaction = torch.tensor([newaction], dtype=torch.float64)

                if actions_reward == env.max_reward: 
                    print("done ", done)
                    winning += 1

                self.get_actions_q(t_action_state, env)[action] += alpha * (actions_reward + gamma * torch.max(self.get_actions_q(t_action_newstate, env)) - self.get_actions_q(t_action_state, env)[action])
                self.get_options_q(t_action_state, t_action, env)[option] += alpha * (options_reward + gamma * torch.max(self.get_options_q(t_action_newstate, t_newaction, env)) - self.get_options_q(t_action_state, t_action, env)[option])

                action_state, action, option = action_newstate, newaction, newoption
                reward = actions_reward + options_reward
                tot_reward += reward
                
            tot_rewards += tot_reward

            if debug and ((m+1)%100 == 0):
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
            print("reward ", actions_reward)
            print("****************************")
        print("sur {nbtests} test l agent a réussi {n} fois".format(nbtests=max_iterations, n=winning))
        
    
    def get_actions_q(self, state, env):
        if state in self.actionsQ:
            return self.actionsQ[state]
        
        temp = torch.zeros(env.action_space.n)
        self.actionsQ[state] = temp
        return temp

    def get_options_q(self, state, action, env):
        option_s = torch.cat((state, action))
        if option_s in self.optionsQ:
            return self.optionsQ[option_s]
        
        temp = torch.zeros(env.options_space.n)
        self.optionsQ[option_s] = temp
        return temp