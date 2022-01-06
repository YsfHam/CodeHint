import torch
from .FuncApproxAgent import FuncApproxAgent
  

class QLearningAgent(FuncApproxAgent):

    def compute_policy(self, env, gamma=0.9, max_iterations=1000000, base_epsilon=0.8, alpha=0.2, debug=False):
        alpha = alpha * self.num_tilings

        tot_rewards = 0
        epsilon = base_epsilon
        if debug:
            print("Training policy...")
        for m in range(max_iterations):
            done = False
            tot_reward = 0.0

            state = env.reset()
            action = self.epsilon_greedy_action(env, state, epsilon)
            infos = None
            reward = 0
            while not done:
                newstate, reward, done, infos = env.step(action)                
                #parameter update
                # if state is final
                # -> .... update (final case)
                # else#
                # -> .... update (non final case)
                newaction = self.epsilon_greedy_action(env, newstate, epsilon)
                stateQ = self.get_q(state, env)
                newstateQ = self.get_q(newstate, env)
                stateFeaturized = self.featurize(state, env)
                if reward == 100: print("bingo")
                if done:
                    self.parameters += alpha * (reward - stateQ[action]) * stateFeaturized[action]
                else:
                    self.parameters += alpha * (reward + gamma * torch.max(newstateQ) - stateQ[action]) * stateFeaturized[action]

                state, action = newstate, newaction
                tot_reward += reward
                
            tot_rewards += tot_reward

            if debug and (((m+1)%100 == 0) or reward == 100):
                avg = tot_rewards / (m+1)
                print("****************************")
                print("success_rate : ", infos['success_rate'])
                print(m+1, avg, epsilon)
                print("------------------------")
                print(env.algorithm)
                print("------------------------")
                print("codage : ", infos['state'])
                print(infos['Errors'])
                print('tests : ', infos['tests'])
                print(infos['actions'])
                print("****************************")
            
            if epsilon > 0:
                epsilon -= base_epsilon/max_iterations
        if debug:
            print("Training finished")
            print("****************************")
            print("success_rate : ", infos['success_rate'])
            print("------------------------")
            print(env.algorithm)
            print("------------------------")
            print("codage : ", infos['state'])
            print(infos['Errors'])
            print("****************************")