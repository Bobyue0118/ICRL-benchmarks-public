import copy
import os
from abc import ABC
from typing import Any, Callable, Dict, Optional, Type, Union

import random
import numpy as np
import torch
from tqdm import tqdm

from common.cns_visualization import traj_visualization_2d
from stable_baselines3.common.dual_variable import DualVariable
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import VecNormalizeWithCost


class PolicyIterationLagrange(ABC):

    def __init__(self,
                 env: Union[GymEnv, str],
                 max_iter: int,
                 n_actions: int,
                 height: int,  # table length
                 width: int,  # table width
                 terminal_states: int,
                 stopping_threshold: float,
                 seed: int,
                 gamma: float = 0.99,
                 v0: float = 0.0,
                 budget: float = 0.,
                 apply_lag: bool = True,
                 penalty_initial_value: float = 1,
                 penalty_learning_rate: float = 0.01,
                 penalty_min_value: Optional[float] = None,
                 penalty_max_value: Optional[float] = None,
                 log_file=None,
                 ):
        super(PolicyIterationLagrange, self).__init__()
        self.stopping_threshold = stopping_threshold
        self.gamma = gamma
        self.env = env
        self.log_file = log_file
        self.max_iter = max_iter
        self.n_actions = n_actions
        self.terminal_states = terminal_states
        self.v0 = v0
        self.seed = seed
        self.height = height
        self.width = width
        self.penalty_initial_value = penalty_initial_value
        self.penalty_min_value = penalty_min_value
        self.penalty_max_value = penalty_max_value
        self.penalty_learning_rate = penalty_learning_rate
        self.apply_lag = apply_lag
        self.budget = budget
        self.num_timesteps = 0
        self.admissible_actions = None
        self.neighbors= [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        self._setup_model()
        self.indicator = 5
        #self.env_for_us = None
        

    def _setup_model(self) -> None:
        self.dual = DualVariable(self.budget,
                                 self.penalty_learning_rate,
                                 self.penalty_initial_value,
                                 min_clamp=self.penalty_min_value,
                                 max_clamp=self.penalty_max_value)
        self.v_m = self.get_init_v()
        self.pi = self.get_equiprobable_policy()

    def get_init_v(self):
        v_m = self.v0 * np.ones((self.height, self.width))
        # # Value function of terminal state must be 0
        # v0[self.e_x, self.e_y] = 0
        return v_m

    # eliminate invalid action
    def get_equiprobable_policy(self):
        pi = 1 / self.n_actions * np.ones((self.height, self.width, self.n_actions))
        for x in range(self.height):
            for y in range(self.width):
                if (x==0 or x==6 or y==0 or y==6) and ([x, y] not in self.terminal_states):
                    for action in range(self.n_actions):
                        next_state = [x+self.neighbors[action][0], y+self.neighbors[action][1]]
                        if not ((0<=next_state[0]<self.height) and (0<=next_state[1]<self.width)):
                            pi[x][y][action] = 0
                    pi[x][y] = pi[x][y] * 1/np.sum(pi[x][y])
        #print('pi',pi) 
        #input('pi')
        return pi
  
    # get current policy
    def get_policy(self):
        return self.pi

    def learn(self,
              #env_for_us,
              total_timesteps: int,
              cost_function: Union[str, Callable],
              latent_info_str: Union[str, Callable] = '',
              callback=None, ):
        #self.env_for_us = env_for_us
        policy_stable, dual_stable = False, False
        iter = 0
        for iter in tqdm(range(total_timesteps)):
            if policy_stable and dual_stable:
                print("\nStable at Iteration {0}.".format(iter), file=self.log_file)
                break
            self.num_timesteps += 1
            # Run the policy evaluation
            self.policy_evaluation(cost_function)
            #print('self.v_m1',self.v_m)
            #input('self.v_m1')
            # Run the policy improvement algorithm
            policy_stable = self.policy_improvement(cost_function)
            cumu_reward, length, dual_stable = self.dual_update(cost_function)
        logger.record("train/iterations", iter)
        logger.record("train/cumulative rewards", cumu_reward)
        logger.record("train/length", length)
        return self.v_m

    # expert learn its value function, Q-value function, thus advantage function
    def expert_learn(self,
              #env_for_us,
              total_timesteps: int,
              cost_function: Union[str, Callable],
              expert_policy: np.ndarray,
              v_m: np.ndarray, 
              latent_info_str: Union[str, Callable] = '',
              transition=None,
              callback=None,):
        #self.env_for_us = env_for_us
        policy_stable, dual_stable = False, False
        iter = 0
        self.pi=expert_policy
        #self.pi = self.get_equiprobable_policy()
        #print('self.pi',np.round(self.pi,2))
        #input('self.pi')
        #for iter in tqdm(range(1)):#need only once policy evaluation
            #if policy_stable and dual_stable:
                #print("\nStable at Iteration {0}.".format(iter), file=self.log_file)
                #break
            #self.num_timesteps += 1
            # Run the policy evaluation
            #self.policy_evaluation(cost_function)
            # Run the policy improvement algorithm
            #policy_stable = self.policy_improvement(cost_function)
            #cumu_reward, length, dual_stable = self.dual_update(cost_function)
        #logger.record("train/iterations", iter)
        #logger.record("train/cumulative rewards", cumu_reward)
        #logger.record("train/length", length)
        self.v_m = v_m
        Q = np.zeros((self.height, self.width, self.n_actions))
        A = np.zeros((self.height, self.width, self.n_actions))
        if transition is not None:
            for i in range(self.height):
                for j in range(self.width):
                    for k in range(self.n_actions):
                        for i1 in range(self.height):
                            for j1 in range(self.width):
                                if transition[i][j][k][i1][j1]!=0:
                                    Q[i][j][k] += self.gamma*transition[i][j][k][i1][j1]*self.v_m[i1][j1]
                        if Q[i][j][k] - self.v_m[i][j] > 0:
                            A[i][j][k] = Q[i][j][k] - self.v_m[i][j] 
        
                    
        return self.v_m, Q, A

    def step(self, action):
        #input('\nstep0')
        return self.env.step(np.asarray([action]))

    def dual_update(self, cost_function):
        """policy rollout for recording training performance"""
        obs = self.env.reset()
        cumu_reward, length = 0, 0
        actions_game, obs_game, costs_game = [], [], []
        while True:
            actions, _ = self.predict(obs=obs, state=None)
            #print('actions',actions)
            actions_game.append(actions[0])
            obs_primes, rewards, dones, infos = self.step(actions)
            if type(cost_function) is str:
                costs = np.array([info.get(cost_function, 0) for info in infos])
                #print('costs',costs)
                if isinstance(self.env, VecNormalizeWithCost):
                    orig_costs = self.env.get_original_cost()
                    #print('orig_costs1',orig_costs),该if成立
                else:
                    orig_costs = costs
                    #print('orig_costs2',orig_costs)
            else:
                costs = cost_function(obs, actions)
                orig_costs = costs
            self.admissible_actions = infos[0]['admissible_actions']
            costs_game.append(orig_costs)
            obs = obs_primes
            obs_game.append(obs[0])
            done = dones[0]
            if done:
                break
            cumu_reward += rewards[0]
            length += 1
        costs_game_mean = np.asarray(costs_game).mean()
        self.dual.update_parameter(torch.tensor(costs_game_mean))
        penalty = self.dual.nu().item()
        print("Performance: dual {0}, cost: {1}, states: {2}, "
              "actions: {3}, rewards: {4}.".format(penalty,
                                                   costs_game_mean.tolist(),
                                                   np.asarray(obs_game).tolist(),
                                                   np.asarray(actions_game).tolist(),
                                                   cumu_reward),
              file=self.log_file,
              flush=True)
        dual_stable = True if costs_game_mean == 0 else False
        return cumu_reward, length, dual_stable

    def policy_evaluation(self, cost_function):
        iter = 0

        delta = self.stopping_threshold + 1
        while delta >= self.stopping_threshold and iter <= self.max_iter-1:
            old_v = self.v_m.copy()
            delta = 0

            # Traverse all states
            for x in range(self.height):
                for y in range(self.width):
                    # Run one iteration of the Bellman update rule for the value function
                    self.bellman_update(old_v, x, y, cost_function)
                    # Compute difference
                    delta = max(delta, abs(old_v[x, y] - self.v_m[x, y]))
            iter += 1
        print("\n\nThe Policy Evaluation algorithm converged after {} iterations".format(iter),
              file=self.log_file)

    def policy_improvement(self, cost_function):
        """Applies the Policy Improvement step."""
        policy_stable = True
        # Iterate states
        for x in range(self.height):
            for y in range(self.width):
                if [x, y] in self.terminal_states:
                    continue
                old_pi = self.pi[x, y, :].copy()

                # Iterate all actions
                action_values = []
                for action in range(self.n_actions):
                    states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
                    assert states[0][0] == x and states[0][1] == y
                    # Compute next state, but not the same
                    # if the action is invalid, continue
                    next_state = [x+self.neighbors[action][0], y+self.neighbors[action][1]]
                    if not ((0<=next_state[0]<self.height) and (0<=next_state[1]<self.width)):
                        action_values.append(-np.inf)
                        continue
                    s_primes, rewards, dones, infos = self.step(action)
                    
                    # Get cost from environment.
                    if type(cost_function) is str:
                        costs = np.array([info.get(cost_function, 0) for info in infos])
                        if isinstance(self.env, VecNormalizeWithCost):
                            orig_costs = self.env.get_original_cost()
                            #第0轮, cost_function全零
                            #if orig_costs[0]!=0:
                            	#input('orig_costs!=0:')
                        else:
                            orig_costs = costs
                    else:
                        costs = cost_function(states, [action])
                        orig_costs = costs
                    current_penalty = self.dual.nu().item()
                    lag_costs = self.apply_lag * current_penalty * orig_costs[0]
                    # Get value
                    curr_val = rewards[0] - lag_costs + self.gamma * self.v_m[s_primes[0][0], s_primes[0][1]]
                    # curr_val = self.v_m[s_primes[0][0], s_primes[0][1]]
                    action_values.append(curr_val)
                #print('action_values',x,y,action_values)
                #input('')
                best_actions = np.argwhere(action_values == np.amax(action_values)).flatten().tolist()
                # Define new policy
                self.define_new_policy(x, y, best_actions)

                # Check whether the policy has changed
                if not (old_pi == self.pi[x, y, :]).all():
                    policy_stable = False

        return policy_stable

    def define_new_policy(self, x, y, best_actions):
        """Defines a new policy given the new best actions.
        Args:
            pi (array): numpy array representing the policy
            x (int): x value position of the current state
            y (int): y value position of the current state
            best_actions (list): list with best actions
            actions (list): list of every possible action
        """

        prob = 1 / len(best_actions)

        for a in range(self.n_actions):
            self.pi[x, y, a] = prob if a in best_actions else 0

    def bellman_update(self, old_v, x, y, cost_function):
        if [x, y] in self.terminal_states:
            return
        total = 0
        for action in range(self.n_actions):
            states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
            assert states[0][0] == x and states[0][1] == y
            # Compute next state, but not the same
            # if the action is invalid, continue
            next_state = [x+self.neighbors[action][0], y+self.neighbors[action][1]]
            if not ((0<=next_state[0]<self.height) and (0<=next_state[1]<self.width)):
                continue

            # Get next state
            s_primes, rewards, dones, infos = self.step(action)
            #print('\nstep:',x,y,action,s_primes, rewards)
            # Get cost from environment.
            if type(cost_function) is str:
                costs = np.array([info.get(cost_function, 0) for info in infos])
                #print('costs',costs)
                #input('Enter...')
                if isinstance(self.env, VecNormalizeWithCost):
                    orig_costs = self.env.get_original_cost()
                    #print('orig_costs',orig_costs)
                    #input('Enter...')
                else:
                    orig_costs = costs
            else:
                costs = cost_function(states, [action])
                orig_costs = costs
            # print(x, y, rewards[0], orig_costs[0])
            gamma_values = self.gamma * old_v[s_primes[0][0], s_primes[0][1]]
            current_penalty = self.dual.nu().item()
            lag_costs = self.apply_lag * current_penalty * orig_costs[0]
            total += self.pi[x, y, action] * (rewards[0] - lag_costs + gamma_values)
        #if x==3 and y in [0,1,2,3,4] and self.indicator>0:
            #print('self.apply_lag * current_penalty',self.apply_lag, current_penalty, orig_costs[0])
            #self.indicator -= 1
            #input('check')
        #print('self.pi',x,y,self.pi)
        #print('cost_function',const_function)
        #input('Enter...')
        self.v_m[x, y] = total
        #if x == 4 and y == 0:
            #print('\n',x,y,self.v_m,'\n\n',np.round(self.pi[x,y],2),self.gamma)
            #input('self.v_m')

    def predict(self, obs, state, deterministic=None):
        policy_prob = copy.copy(self.pi[int(obs[0][0]), int(obs[0][1])])
        if self.admissible_actions is not None:
            for c_a in range(self.n_actions):
                if c_a not in self.admissible_actions:
                    policy_prob[c_a] = -float('inf')
        #print('policy_prob',obs[0][0],obs[0][1],policy_prob,self.admissible_actions)
        #input('Enter')
        best_actions = np.argwhere(policy_prob == np.amax(policy_prob)).flatten().tolist()
        action = random.choice(best_actions)
        #print('action',action)
        return np.asarray([action]), state

    def predict_random(self, obs, state, deterministic=None):
        """
        predict the action randomly
        """
        policy_prob = copy.copy(self.pi[int(obs[0][0]), int(obs[0][1])])
        if self.admissible_actions is not None:
            for c_a in range(self.n_actions):
                if c_a not in self.admissible_actions:
                    policy_prob[c_a] = -float('inf')
        print('policy_prob',obs[0][0],obs[0][1],policy_prob,self.admissible_actions)
        input('Enter')
        action = random.choice([i for i in range(len(policy_prob)) if policy_prob[i]>0 or policy_prob[i]==0]) #us-code,随机选择一个合理的行动
        print('action',action)
        return np.asarray([action]), state

    def save(self, save_path):
        state_dict = dict(
            pi=self.pi,
            v_m=self.v_m,
            gamma=self.gamma,
            max_iter=self.max_iter,
            n_actions=self.n_actions,
            terminal_states=self.terminal_states,
            seed=self.seed,
            height=self.height,
            width=self.width,
            budget=self.budget,
            num_timesteps=self.num_timesteps,
            stopping_threshold=self.stopping_threshold,
        )
        torch.save(state_dict, save_path)


def load_pi(model_path, iter_msg, log_file):
    if iter_msg == 'best':
        model_path = os.path.join(model_path, "best_nominal_model")
    else:
        model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'nominal_agent')
    print('Loading model from {0}'.format(model_path), flush=True, file=log_file)

    state_dict = torch.load(model_path)

    pi = state_dict["pi"]
    v_m = state_dict["v_m"]
    gamma = state_dict["gamma"]
    max_iter = state_dict["max_iter"]
    n_actions = state_dict["n_actions"]
    terminal_states = state_dict["terminal_states"]
    seed = state_dict["seed"]
    height = state_dict["height"]
    width = state_dict["width"]
    budget = state_dict["budget"]
    stopping_threshold = state_dict["stopping_threshold"]

    create_iteration_agent = lambda: PolicyIterationLagrange(
        env=None,
        max_iter=max_iter,
        n_actions=n_actions,
        height=height,  # table length
        width=width,  # table width
        terminal_states=terminal_states,
        stopping_threshold=stopping_threshold,
        seed=seed,
        gamma=gamma,
        budget=budget, )
    iteration_agent = create_iteration_agent()
    iteration_agent.pi = pi
    iteration_agent.v_m = v_m

    return iteration_agent