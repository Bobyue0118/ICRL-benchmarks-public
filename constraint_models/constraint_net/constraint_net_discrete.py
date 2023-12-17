from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
import numpy as np
import torch as th
from torch import nn
import copy

class ConstraintDiscrete(nn.Module):
    def __init__(
            self,
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            task: str = 'ICRL',
            env_configs: dict = None,
            device: str = "cpu",
            log_file=None,
            **kwargs,
    ):
        super(ConstraintDiscrete, self).__init__()
        self.task = task
        self.env_configs = env_configs
        self.expert_obs = expert_obs
        self.expert_acs = expert_acs
        self.n_actions = env_configs['n_actions']
        self.transition_prob = self.env_configs['transition_prob']
        self.h = self.env_configs['map_height']
        self.w = self.env_configs['map_width']
        if self.n_actions == 9:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)]  # effect of each movement
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            #self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu', 8: 's'}
            #self.action_space = gym.spaces.Discrete(9)
        elif self.n_actions == 8:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]  # effect of each movement
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7]
            #self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu'}
            #self.action_space = gym.spaces.Discrete(8)
        elif self.n_actions == 4:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # effect of each movement
            self.actions = [0, 1, 2, 3]
            #self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u'}
            #self.action_space = gym.spaces.Discrete(4)
        else:
            raise EnvironmentError("Unknown number of actions {0}.".format(n_actions))
        self.terminals = self.env_configs['terminal_states']
        self._build()

    def _build(self) -> None:
        self.cost_matrix = np.zeros([self.env_configs['map_height'], self.env_configs['map_width']])
        self.cost_matrix_weight = np.zeros([self.env_configs['map_height'], self.env_configs['map_width']])
        self.cost_matrix_sa = np.zeros([self.env_configs['map_height'], self.env_configs['map_width'], self.env_configs['n_actions']])
        self.cost_matrix_zero = np.zeros([self.env_configs['map_height'], self.env_configs['map_width']])
        self.expert_policy_matrix = np.zeros([self.env_configs['map_height'], self.env_configs['map_width'], self.env_configs['n_actions']])
        self.expert_policy_matrix_copy = np.zeros([self.env_configs['map_height'], self.env_configs['map_width'], self.env_configs['n_actions']])
        self.recon_obs = False

    def cost_function(self, obs: np.ndarray, acs: np.ndarray, force_mode: str = None) -> np.ndarray:
        cost = []
        for i in range(obs.shape[0]):
            cost.append(self.cost_matrix[int(obs[i, 0])][int(obs[i, 1])])
        #print('obs',obs)
        #print('self.cost_matrix',self.cost_matrix)
        #print('cost',cost)
        #print('np.asarray(cost)',np.asarray(cost),obs.shape[0])
        #input('Enter...')
        return np.asarray(cost)

    def cost_function_zero(self, obs: np.ndarray, acs: np.ndarray, force_mode: str = None) -> np.ndarray:
        cost = []
        for i in range(obs.shape[0]):
            cost.append(self.cost_matrix_zero[int(obs[i, 0])][int(obs[i, 1])])
        #print('obs',obs)
        #print('self.cost_matrix',self.cost_matrix)
        #print('cost',cost)
        #print('np.asarray(cost)',np.asarray(cost),obs.shape[0])
        #input('Enter...')
        return np.asarray(cost)    

    def expert_policy(self):
        """
        obtain expert policy from expert data
        """
        for i in range(len(self.expert_obs)):#有多少条专家轨迹
            for j in range(len(self.expert_obs[i])):#每条专家轨迹多长
                self.expert_policy_matrix[self.expert_obs[i][j][0],self.expert_obs[i][j][1],self.expert_acs[i][j]] += 1
        print('self.expert_policy_matrix',self.expert_policy_matrix)
        input('expert_policy')

        for i in range(self.env_configs['map_height']):
            for j in range(self.env_configs['map_width']):
                for k in range(self.env_configs['n_actions']):
                    self.expert_policy_matrix_copy[i, j, k]=self.expert_policy_matrix[i, j, k]/max(np.sum(self.expert_policy_matrix[i][j]),1)
        return self.expert_policy_matrix_copy

    def terminal(self, state):
        """
        Check if the state is terminal.
        """
        for terminal_state in self.terminals:
            if state[0] == terminal_state[0] and state[1] == terminal_state[1]:
                return True
        return False

    def get_next_states_and_probs(self, state, action):
        """
        Given a state and action, return list of (next_state, probability) pairs.
        """
        if self.terminal(state):
            return [((state[0], state[1]), 1)]
        if self.transition_prob == 1:
            inc = self.neighbors[action]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if 0 <= nei_s[0] < self.h and \
                    0 <= nei_s[1] < self.w:
                return [(nei_s, 1)]
            else:
                return [((state[0], state[1]), 1)]  # state invalid
        else:
            mov_probs = np.zeros([self.n_actions])
            mov_probs[action] = self.transition_prob
            mov_probs += (1 - self.transition_prob) / self.n_actions
            for a in range(self.n_actions):
                inc = self.neighbors[a]
                nei_s = (state[0] + inc[0], state[1] + inc[1])
                if nei_s[0] < 0 or nei_s[0] >= self.h or \
                        nei_s[1] < 0 or nei_s[1] >= self.w:
                    # mov_probs[-1] += mov_probs[a]
                    mov_probs[a] = 0
            # sample_action = random.choices([i for i in range(self.n_actions)], weights=mov_probs, k=1)[0]
            # inc = self.neighbors[sample_action]
            # return [((state[0] + inc[0], state[1] + inc[1]), 1)]
            res = []
            mov_probs = mov_probs * 1/np.sum(mov_probs)
            for a in range(self.n_actions):
                if mov_probs[a] != 0:
                    inc = self.neighbors[a]
                    nei_s = (state[0] + inc[0], state[1] + inc[1])
                    res.append((nei_s, mov_probs[a]))
            return res


    def train_traj_nn(
            self,
            nominal_obs: np.ndarray,
            **kwargs
    ) -> Dict[str, Any]:
        zeta = 1 #按照论文中，是一个大于\epsilon的数
        A_min = 0 #由于transition model是stochastic，所以可以加入一个阈值，减少因为stochastic导致的A>0
        A = np.round(kwargs['advantage_function'],2) # 防止某些明明是0，但是因为精度，比0大一点的也被下面算进去
        #print('A:',A)
        min_A = np.inf 
        # 防止有误，重新更零。但随着transition model的精确，以及V(s)的精确，最终一定会精确的
        self.cost_matrix = np.zeros([self.env_configs['map_height'], self.env_configs['map_width']])
        self.cost_matrix_sa = np.zeros([self.env_configs['map_height'], self.env_configs['map_width'], self.env_configs['n_actions']])
        #self.cost_matrix_weight = np.zeros([self.env_configs['map_height'], self.env_configs['map_width']])
        
        # from c(s,a) to c(s)
        for i in range(self.h):
            for j in range(self.w):
                for k in range(self.n_actions):
                    res = self.get_next_states_and_probs([i,j],k)
                    if A[i][j][k] > A_min: 
                        self.cost_matrix_sa[i][j][k] = A[i][j][k]*zeta
                        if A[i][j][k] < min_A:
                            min_A = A[i][j][k]
                        for m in range(len(res)):
                            next_state = res[m][0]
                            mov_probs = res[m][1]
                            self.cost_matrix[next_state[0]][next_state[1]] += mov_probs * self.cost_matrix_sa[i][j][k]
        self.cost_matrix /= min_A # 除以大于零的最小的A
        self.cost_matrix[self.cost_matrix<0.04*np.amax(self.cost_matrix)]=0 # 有些因为概率导致的cost略大于0可以删去
        #self.cost_matrix[self.cost_matrix>=0.04*np.amax(self.cost_matrix)]=1
        # Prepare data
        #nominal_obs = np.concatenate(nominal_obs, axis=0)
        #expert_obs = np.concatenate(self.expert_obs, axis=0)
        #print('nominal_obs',nominal_obs)
        #print('expert_obs',self.expert_obs[0],self.expert_acs[0])
        #input('Enter...')
        #for i in range(len(nominal_obs)):
            #is_in = False
            #for j in range(len(expert_obs)):
                #if np.array_equal(nominal_obs[i], expert_obs[j]):
                    #is_in = True
                    #break
            #if is_in == False:
                #self.cost_matrix[nominal_obs[i][0]][nominal_obs[i][1]] = 1
        np.set_printoptions(suppress=True)
        print('self.cost_matrix\n',np.round(self.cost_matrix,2))
        bw_metrics = {"backward/test": 'True'}

        return bw_metrics

    def save(self, save_path):
        state_dict = dict(
            matrix=self.cost_matrix,
        )
        th.save(state_dict, save_path)
