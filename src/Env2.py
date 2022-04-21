import statement
import myExceptions

from gym.spaces import Discrete
import random
import torch

NoneType = type(None)

# Pour le nouvel environement
class Env2:
    def __init__(self, horizon=30):
        self.nb_steps = 0
        self.horizon = horizon

        self.random = random.Random()

        self.reset()
        self.best_state = (0) * self.state_len
        self.max_reward = 100

        self.actions_str = {}
        self.actions_str[(0, 0)] =  'ajouter poser'
        self.actions_str[(0, 1)] =  'ajouter retirer'
        self.actions_str[(0, 2)] =  'ajouter une condition est_vide'
        self.actions_str[(0, 3)] =  'ajouter un block Sinon'
        self.actions_str[(1, 0)] =  'appliquer négation'
        self.actions_str[(2, 0)] =  'enlever arbre gauche'
        self.actions_str[(2, 1)] =  'enlever arbre droit'
        self.actions_str[(3, 0)] =  'Blank'

        self.actions = [x for x in self.actions_str]
        self.action_space = Discrete(len(self.actions))

        self.possible_actions = {}
        self.possible_actions[statement.StatementsBlock] = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1)]
        self.possible_actions[statement.ConditionStatement] = [(0, 3), (1, 0), (2, 1)]
        self.possible_actions[statement.FunctionStatement] = [(3, 0)]


        self.prepared_add_statements = [0] * len([x for x, _ in self.actions_str if x == 0])
        self.prepared_add_statements[0] = lambda:statement.FunctionStatement('poser', 2, lambda x: x + 1)
        self.prepared_add_statements[1] = lambda:statement.FunctionStatement('retirer', 3, lambda x: x - 1)
        self.prepared_add_statements[2] = lambda:statement.ConditionStatement('est_vide', 4, lambda x:x == 0)
        self.prepared_add_statements[3] = lambda:statement.StatementsBlock()

        options_number = 7
        # création d'une liste contenant une chaine de caractère vide option_number fois (une sorte d'allocation de la mémoire)
        self.options_str = [''] * options_number 
        self.options_str[0] = 'Aller à gauche'
        self.options_str[1] = 'Aller à droite'
        self.options_str[2] = 'Aller au prochain bloc à gauche'
        self.options_str[3] = 'Aller au prochain bloc à droite'
        self.options_str[4] = 'Nop'
        self.options_str[5] = 'Monter'
        self.options_str[6] = 'Monter au bloc precedent'

        self.options_space = Discrete(options_number)

        self.action_possible_options = {}
        self.action_possible_options[(0, 0)] = [1, 3, 5, 6, 4]
        self.action_possible_options[(0, 1)] = [1, 3, 5, 6, 4]
        self.action_possible_options[(0, 2)] = [0, 2, 5, 6, 4]
        self.action_possible_options[(0, 3)] = [1, 3, 5, 6, 4]
        self.action_possible_options[(1, 0)] = [0, 1, 2, 3, 5, 6, 4]
        self.action_possible_options[(2, 0)] = [3, 1, 5, 6, 4]
        self.action_possible_options[(2, 1)] = [2, 0, 5, 6, 4]
        self.action_possible_options[(3, 0)] = [5, 6]

        self.statementsCodes = {
            statement.StatementsBlock:0,
            statement.FunctionStatement:1,
            statement.ConditionStatement:1
        }

    def print(self):
        #print(self.algorithm)
        print(self.infos)
        print(self.state_infos)

    def sample(self):
        action_index = self.action_space.sample()
        action = self.actions[action_index]
        options = self.action_possible_options[action]
        return action_index, self.random.choice(options)

    def reset(self):
        self.nb_steps = 0
        self.algorithm = statement.StatementsBlock()
        self.current_statement = self.algorithm

        self.reward = 0
        self.infos = {
            'actions':[],
            'success_rate':0,
            'algo_results':[],
            'algo':'empty',
            'codage':self.algorithm.encode_str()
        }

        self.state_infos = {
            'codage': self.algorithm.encode_str(),
            'tests':0,
            'current_statement':0, # 0 pour un bloc d'instruction
            'actions_option':4
        }

        self.state_len = len(self.state_infos)
        state = [self.state_infos[x] for x in self.state_infos]
        state[0] = self.algorithm.encode()
        state = tuple(state)
        return state

    def step(self, actionIndex, option):
        if not self.action_space.contains(actionIndex):
            raise Exception("Action " + actionIndex + " not found in actions space")
        if not self.options_space.contains(option):
            raise Exception("Option " + option + " not found in options space")

        self.nb_steps += 1

        action = self.actions[actionIndex]

        done = False
        options_reward = -1
       
        if self.is_action_valid(action) and self.is_option_valid(action, option):
            options_reward = 1
            self.perform_action(action)
            self.perform_move_option(option)

            self.infos['actions'].append(
                (self.actions_str[action], self.options_str[option])
            )

            self.state_infos['current_statement'] = self.statementsCodes[type(self.current_statement)]
            self.state_infos['actions_option'] = option

            self.state_infos['codage'] = self.infos['codage'] = self.algorithm.encode_str()
            self.infos['algo'] = str(self.algorithm)
            
        done = self.test_algorithm()
        self.reward -= self.algorithm.instructionsNum()
        #retun new state
        state = [self.state_infos[x] for x in self.state_infos]
        state[0] = self.algorithm.encode()
        state = tuple(state)
        if self.reward == self.max_reward:
            self.best_state = state
        '''else:
            self.reward -= torch.abs(torch.tensor(state, dtype=torch.float64) - torch.tensor(self.best_state, dtype=torch.float64)).sum().item()
        '''
        return state, self.reward, options_reward, done or self.nb_steps == self.horizon, self.infos
    
    def test_algorithm(self):
        self.infos['algo_results'] = []
        bits = ''
        major_error = False
        success = 0
        self.state_infos['tests'] = 0
        test_io = [(0, 0), (1, 0)]
        try:
            for input, output in test_io:
                res = self.algorithm.evaluate(input)
                self.infos['algo_results'].append((input, res))
                if res == output: 
                    self.reward += 1
                    bits = bits + '1'
                    success += 1
                else:
                    self.reward -= 1
                    bits = bits + '0'
                
                if res < 0:
                    major_error = True
            self.state_infos['tests'] = int(bits, 2)
        except statement.AgentError as e:
            major_error = True
        
        self.infos['success_rate'] = success / len(test_io) * 100
        if major_error: 
            self.reward = -self.max_reward * 0.5
            return False
        if success == len(test_io):
            self.reward = self.max_reward
            return True
        return False

    def is_action_valid(self, action):
        if action not in self.possible_actions[type(self.current_statement)]:
            return False
        action_class, action_code = action
        if action_class == 2 and action_code == 0:
            return self.current_statement.getLeft() is not None
        elif action_class == 2 and action_code == 1:
            return self.current_statement.getRight() is not None
        
        return True

    def perform_action(self, action):
        action_class, action_code = action
        if action_class == 0:
            self.current_statement.add(self.prepared_add_statements[action_code]())
        elif action_class == 1:
            self.current_statement.negation()
        elif action_class == 2:
            self.current_statement.setLeft(None) if action_code == 0 else self.current_statement.setRight(None)

    def is_option_valid(self, action, option):
        if option not in self.action_possible_options[action]:
            return False
        if option == 0:
            return self.current_statement.getLeft() is not None and type(self.current_statement.getLeft()) != statement.FunctionStatement
        elif option == 2:
            if self.current_statement.getLeft() is None:
                return False
            temp = self.current_statement.getLeft()
            while type(temp) not in [NoneType, statement.StatementsBlock]:
                temp = temp.getLeft()
            return temp is not None
        elif option in [1, 3]:
            if self.current_statement.getRight() is None:
                return False
            temp = self.current_statement.getRight()
            while type(temp) not in [NoneType, statement.StatementsBlock]:
                temp = temp.getRight()
            return temp is not None
        elif option == 4:
            return True
        elif option in [5, 6]:
            return self.current_statement.parent is not None
        return False
    def perform_move_option(self, option):
        if option == 0:
            self.current_statement = self.current_statement.getLeft()
        elif option == 1:
            self.current_statement = self.current_statement.getRight()
        elif option == 2:
            self.current_statement = self.current_statement.getLeft()
            while type(self.current_statement) != statement.StatementsBlock:
                self.current_statement = self.current_statement.getLeft()
        elif option == 3:
            self.current_statement = self.current_statement.getRight()
            while type(self.current_statement) != statement.StatementsBlock:
                self.current_statement = self.current_statement.getRight()
        elif option == 4:
            return
        elif option == 5:
            self.current_statement = self.current_statement.parent
        elif option == 6:
            self.current_statement = self.current_statement.parent
            while type(self.current_statement) != statement.StatementsBlock:
                self.current_statement = self.current_statement.parent