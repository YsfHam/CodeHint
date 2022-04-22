from .FuncApproxAgent import FuncApproxAgent
import torch
class MonteCarloAgent(FuncApproxAgent):
     def compute_policy(self, env, gamma=0.9, max_iterations=1000000, base_epsilon=0.8, alpha=0.2, debug=False):

        tot_rewards = 0
        epsilon = base_epsilon

        if debug:
            print("Training policy...")
        for m in range(max_iterations):
            done = False
            a_tot_reward = 0.0
            o_tot_reward = 0

            action_state = env.reset()
            t_action_state = torch.tensor(action_state, dtype=torch.float64)
            action, option = self.epsilon_greedy_action(env, t_action_state, epsilon)
            infos = None
            states_actions = [(action_state, action, option)]
            reward = 0
            while not done:
                action_state, a_reward, o_reward, done, infos = env.step(action, option)                
                action, option = self.epsilon_greedy_action(env, t_action_state, epsilon)
                states_actions.append((action_state, action, option))
                a_tot_reward += a_reward
                o_tot_reward += o_reward

            for action_state, action, option in states_actions:
                t_action_state = torch.tensor(action_state, dtype=torch.float64)
                t_action = torch.tensor([action], dtype=torch.float64)
                self.actions_parameters += alpha * (a_tot_reward - self.get_actions_q(t_action_state, env)[action]) * t_action_state
                self.options_parameters += alpha * (o_tot_reward - self.get_options_q(t_action_state, t_action, env)[option]) * torch.cat((t_action_state, t_action))
                
            tot_rewards += a_tot_reward + o_tot_reward

            if debug and (((m+1)%100 == 0) or reward == 100):
                avg = tot_rewards / (m+1)
                print(m+1, avg, epsilon)
                print(env.algorithm)
            
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
