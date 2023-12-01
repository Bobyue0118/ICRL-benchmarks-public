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

        self._build()

    def _build(self) -> None:
        self.cost_matrix = np.zeros([self.env_configs['map_height'], self.env_configs['map_width']])
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

    def expert_policy(self):
        """
        obtain expert policy from expert data
        """
        for i in range(len(self.expert_obs)):
            for j in range(len(self.expert_obs[i])):
                self.expert_policy_matrix[self.expert_obs[i][j][0],self.expert_obs[i][j][1],self.expert_acs[i][j]] += 1
        print('self.expert_policy_matrix',self.expert_policy_matrix)

        for i in range(self.env_configs['map_height']):
            for j in range(self.env_configs['map_width']):
                for k in range(self.env_configs['n_actions']):
                    self.expert_policy_matrix_copy[i, j, k]=self.expert_policy_matrix[i, j, k]/max(np.sum(self.expert_policy_matrix[i][j]),1)
        return self.expert_policy_matrix_copy


    def train_traj_nn(
            self,
            nominal_obs: np.ndarray,
            **kwargs
    ) -> Dict[str, Any]:
        # Prepare data
        nominal_obs = np.concatenate(nominal_obs, axis=0)
        expert_obs = np.concatenate(self.expert_obs, axis=0)
        #print('nominal_obs',nominal_obs)
        #print('expert_obs',self.expert_obs[0],self.expert_acs[0])
        #input('Enter...')
        for i in range(len(nominal_obs)):
            is_in = False
            for j in range(len(expert_obs)):
                if np.array_equal(nominal_obs[i], expert_obs[j]):
                    is_in = True
                    break
            if is_in == False:
                self.cost_matrix[nominal_obs[i][0]][nominal_obs[i][1]] = 1
        print('self.cost_matrix',self.cost_matrix)
        bw_metrics = {"backward/test": 'True'}
        #print('self.expert_policy_matrix1',self.expert_policy())
        #input('enter0001')
        return bw_metrics

    def save(self, save_path):
        state_dict = dict(
            matrix=self.cost_matrix,
        )
        th.save(state_dict, save_path)
