import torch
from .FuncApproxAgent import FuncApproxAgent
  

class QLearningAgent(FuncApproxAgent):

    def __init__(self, env):
        super().__init__(env)

    def compute_policy(self, env, gamma=0.9, max_iterations=1000000, base_epsilon=0.8, alpha=0.2, debug=False):

        tot_rewards = 0
        epsilon = base_epsilon
        if debug:
            print("Training policy...")

        for m in range(max_iterations):
            done = False
            tot_reward = 0.0

            action_state = env.reset()
            t_action_state = torch.tensor(action_state, dtype=torch.float)
            action, option = self.epsilon_greedy_action(env, t_action_state, epsilon)
            infos = None
            reward = 0
            while not done:
                t_action = torch.tensor([action], dtype=torch.float)
                t_option_state = torch.cat((t_action_state, t_action))
                action_newstate, reward, done, infos = env.step(action, option)                
                t_action_newstate = torch.tensor(action_newstate, dtype=torch.float)
                newaction, newoption = self.epsilon_greedy_action(env, t_action_newstate, epsilon)
                stateActionQ = self.get_actions_q(t_action_state, env)
                stateOptionQ = self.get_options_q(t_action_state,t_action, env)
                newstateActionQ = self.get_actions_q(t_action_newstate, env)
                t_newaction = torch.tensor([newaction], dtype=torch.float)
                newstateOptionQ = self.get_options_q(t_action_newstate, t_newaction, env)
                if reward == 100: print("bingo")
                if done and reward == 100:
                    self.actions_parameters += alpha * (reward - stateActionQ[action]) * t_action_state
                    self.options_parameters += alpha * (reward - stateOptionQ[option]) * t_option_state
                else:
                    self.actions_parameters += alpha * (reward + gamma * torch.max(newstateActionQ) - stateActionQ[action]) * t_action_state
                    self.options_parameters += alpha * (reward + gamma * torch.max(newstateOptionQ) - stateOptionQ[option]) * t_option_state

                action_state, action, option = action_newstate, newaction, newoption
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
            print("------------------------")
            print(self.actions_parameters)
            print("------------------------")
            print(self.options_parameters)
            print("****************************")
        self.save_to_file(["actions.pol", "options.pol"])