import numpy as np


"""Implements the two-timescale stochastic approximation"""

def cal_gra_of_x(lambda_1, lambda_2, ci, cost_k, reward, env):
    gra_of_x = -ci+lambda_1*cost_k-lambda_2*reward
    # gradient of invalid (s,a) equals to zero
    for i in range(env.h):
            for j in range(env.w):
                if [i, j] in env.terminals:
                    gra_of_x[i][j] = 0
                if (i==0 or i==6 or j==0 or j==6) and ([i, j] not in env.terminals):
                    for action in range(env.n_actions):
                        next_state = [i+env.neighbors[action][0], j+env.neighbors[action][1]]
                        if not ((0<=next_state[0]<env.h) and (0<=next_state[1]<env.w)):
                            gra_of_x[i][j][action] = 0
    return gra_of_x

def cal_gra_of_lambda_1(gamma, v_c, vareps_k, eps, x, cost_k):
    #print(v_c,vareps_k,np.sum(x*cost_k))
    #input('inside gra_of_lambda_1')
    return -(1-gamma)*(v_c+4*vareps_k+2*eps)+np.sum(x*cost_k)

def cal_gra_of_lambda_2(gamma, v_r, R_k, x, reward):
    return (1-gamma)*(v_r+R_k)-np.sum(x*reward)

def update_x(x, gra_of_x, a_k):
    y=x-a_k*gra_of_x
    y[y<0]=0         # probability is non-negative
    y /= np.sum(y)   # probability should be normalized
    return y

def update_lambda_1(lambda_1, gra_of_lambda_1, b_k):
    return lambda_1+b_k*gra_of_lambda_1

def update_lambda_2(lambda_2, gra_of_lambda_2, b_k):
    return lambda_2+b_k*gra_of_lambda_2

def cal_R_k(gamma, transition, estimated_transition, expert_policy, expert_policy_active, R_max = 1):
    #print(np.max(abs(transition-estimated_transition)), np.max(abs(expert_policy-expert_policy_active)))
    #input('inside')
    R_k = 2*gamma*R_max*np.max(abs(transition-estimated_transition))/(1-gamma)**2 + gamma*R_max*np.max(abs(expert_policy-expert_policy_active))/(1-gamma)**2
    return R_k

def cal_pi_expl(height, width, n_actions, x_k, env, k):
    #if k == 1:
        #return env.get_equiprobable_policy()
    pi_k = np.zeros((height, width, n_actions))
    for i in range(height):
        for j in range(width):
            sum_of_acs = 0
            for action in env.get_actions([i,j]): #只对valid action求和
                sum_of_acs += x_k[i,j,action]
            for action in env.get_actions([i,j]):
                pi_k[i,j,action] = x_k[i,j,action]/sum_of_acs
    return pi_k


