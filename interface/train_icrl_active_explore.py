import datetime
import json
import os
import sys
import time
import gym
import numpy as np
import yaml
from copy import copy, deepcopy

cwd = os.getcwd()#返回当前工作路径(CWD: Current Working Directory),是脚本运行的地方
sys.path.append(cwd.replace('/interface', ''))
cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', '/mujuco_environment'))
from utils.true_constraint_functions import get_true_cost_function
from stable_baselines3.iteration.policy_interation_lag_discrete import PolicyIterationLagrange
from common.cns_sampler import ConstrainedRLSampler
from common.cns_evaluation import evaluate_icrl_policy
from common.cns_visualization import constraint_visualization_1d, constraint_visualization_2d, traj_visualization_2d
from common.cns_env import make_train_env, make_eval_env
from common.memory_buffer import IRLDataQueue
from constraint_models.constraint_net.variational_constraint_net import VariationalConstraintNet
from constraint_models.constraint_net.constraint_net import ConstraintNet
from constraint_models.constraint_net.constraint_net_discrete import ConstraintDiscrete
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common import logger

from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize
from utils.data_utils import read_args, load_config, ProgressBarManager, del_and_make, load_expert_data, \
    get_input_features_dim, process_memory, print_resource
from utils.model_utils import load_ppo_config, load_policy_iteration_config, get_hoeffding_ci_us, get_hoeffding_ci_active, costValueIteration, cal_GIoU, cal_discounted_cumulative_rewards, cal_discounted_cumulative_costs
from utils.optimization import cal_gra_of_x, cal_gra_of_lambda_1, cal_gra_of_lambda_2, cal_R_k, update_x, update_lambda_1, update_lambda_2, cal_pi_expl

import warnings  # disable warnings

from mujuco_environment.custom_envs.envs.wall_gird_word import WallGridworld
warnings.filterwarnings("ignore")# 忽略匹配的警告


def null_cost(x, *args):
    # Zero cost everywhere
    return np.zeros(x.shape[:1])#如果a=[[1,2],[3,4]],那么a[:1]=[[1,2]]，包含最外侧的括号


def train(config):
    # load config
    #print('args inside the function are:',args)
    #print('config inside the function are:',config)
    config, debug_mode, log_file_path, partial_data, num_threads, seed = load_config(args)# 定义在utils/data_utils.py
    if num_threads > 1:
        multi_env = True
        config.update({'multi_env': True})
    else:
        multi_env = False
        config.update({'multi_env': False})

    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    debug_msg = ''
    if debug_mode:  # this is for a local debugging, use python train_icrl.py -d 1 to enable the debug model
        config['device'] = 'cpu'
        config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        config['running']['n_eval_episodes'] = 10
        config['running']['save_every'] = 1
        if 'PPO' in config.keys():
            config['PPO']['forward_timesteps'] = 3000  # 2000
            config['PPO']['n_steps'] = 500
            config['PPO']['n_epochs'] = 2
            config['running']['sample_rollouts'] = 2
            config['CN']['backward_iters'] = 2
        elif 'iteration' in config.keys():
            config['iteration']['max_iter'] = 2
        # config['CN']['cn_batch_size'] = 1000
        debug_msg = 'debug-'

    # print the current config
    print(json.dumps(config, indent=4), file=log_file, flush=True)

    # init data buffer
    if config['running']['use_buffer']:
        sample_data_queue = IRLDataQueue(max_rollouts=config['running']['store_sample_rollouts'],
                                         store_by_game=config['running']['store_by_game'],
                                         seed=seed)

    # init saving dir for the running models
    current_time_date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M')
    save_model_mother_dir = '{0}/{1}/{5}{2}{3}-{4}-seed_{6}/'.format(
        config['env']['save_dir'],
        config['task'],
        args.config_file.split('/')[-1].split('.')[0],
        '-multi_env' if multi_env else '',
        current_time_date,
        debug_msg,
        seed
    )
    if not os.path.exists('{0}/{1}/'.format(config['env']['save_dir'], config['task'])):
        os.mkdir('{0}/{1}/'.format(config['env']['save_dir'], config['task']))
    if not os.path.exists(save_model_mother_dir):
        os.mkdir(save_model_mother_dir)
    if not os.path.exists(save_model_mother_dir+ '/cost_matrix_estimated'):
        os.mkdir(save_model_mother_dir+ '/cost_matrix_estimated')
    print("Saving to the file: {0}".format(save_model_mother_dir), file=log_file, flush=True)
    # save the running config
    with open(os.path.join(save_model_mother_dir, "model_hyperparameters.yaml"), "w") as hyperparam_file:
        yaml.dump(config, hyperparam_file)

    # monitor the memory and running time
    mem_prev = process_memory()
    time_prev = time.time()

    # init training env
    train_env, env_configs = make_train_env(env_id=config['env']['train_env_id'],
                                            config_path=config['env']['config_path'],
                                            save_dir=save_model_mother_dir,
                                            group=config['group'],
                                            base_seed=seed,
                                            num_threads=num_threads,
                                            use_cost=config['env']['use_cost'],
                                            normalize_obs=not config['env']['dont_normalize_obs'],
                                            normalize_reward=not config['env']['dont_normalize_reward'],
                                            normalize_cost=not config['env']['dont_normalize_cost'],
                                            cost_info_str=config['env']['cost_info_str'],
                                            reward_gamma=config['env']['reward_gamma'],
                                            cost_gamma=config['env']['cost_gamma'],
                                            multi_env=multi_env,
                                            part_data=partial_data,
                                            log_file=log_file,
                                            noise_mean=config['env']['noise_mean'] if 'Noise' in config['env'][
                                                'train_env_id'] else None,
                                            noise_std=config['env']['noise_std'] if 'Noise' in config['env'][
                                                'train_env_id'] else None,
                                            max_scene_per_env=config['env']['max_scene_per_env']
                                            if 'max_scene_per_env' in config['env'].keys() else None,
                                            )

    # init sample env
    save_valid_mother_dir = os.path.join(save_model_mother_dir, "sample/")
    if not os.path.exists(save_valid_mother_dir):
        os.mkdir(save_valid_mother_dir)
    if 'sample_multi_env' in config['running'].keys() and config['running']['sample_multi_env'] and num_threads > 1:
        # if require multi-process sampling
        sample_multi_env = True
        sample_num_threads = num_threads
    else:
        sample_multi_env = False
        sample_num_threads = 1
    sampling_env, env_configs = make_eval_env(env_id=config['env']['train_env_id'],
                                              config_path=config['env']['config_path'],
                                              save_dir=save_valid_mother_dir,
                                              group=config['group'],
                                              num_threads=sample_num_threads,
                                              mode='sample',
                                              use_cost=config['env']['use_cost'],
                                              normalize_obs=not config['env']['dont_normalize_obs'],
                                              cost_info_str=config['env']['cost_info_str'],
                                              part_data=partial_data,
                                              multi_env=sample_multi_env,
                                              log_file=log_file,
                                              noise_mean=config['env']['noise_mean'] if 'Noise' in config['env'][
                                                  'train_env_id'] else None,
                                              noise_std=config['env']['noise_std'] if 'Noise' in config['env'][
                                                  'train_env_id'] else None,
                                              circle_info=config['env']['circle_info'] if 'Circle' in config[
                                                  'env']['train_env_id'] else None,
                                              max_scene_per_env=config['env']['max_scene_per_env']
                                              if 'max_scene_per_env' in config['env'].keys() else None
                                              )
    if "planning" in config['running'].keys() and config['running']['planning']:
        # if require planner (cross-entropy planner and tree planner) for sampling
        planning_config = config['Plan']
        config['Plan']['top_candidates'] = int(config['running']['sample_rollouts'])
    else:
        planning_config = None

    # init active sampling env
    env_configs_copy = copy(env_configs)
    env_active = gym.make(id=config['env']['train_env_id'], **env_configs_copy)
    


    # init sampler
    sampler = ConstrainedRLSampler(rollouts=int(config['running']['sample_rollouts']),
                                   store_by_game=True,  # I move the step out
                                   cost_info_str=config['env']['cost_info_str'],
                                   sample_multi_env=sample_multi_env,
                                   env_id=config['env']['eval_env_id'],
                                   env=sampling_env,
                                   planning_config=planning_config)

    # monitor the memory and running time
    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Loading environment', log_file=log_file)

    # Set obs specs
    recon_obs = config['CN']['recon_obs'] if 'recon_obs' in config['CN'].keys() else False
    if recon_obs:  # mainly for the grid-world env, using one-hot to represent (x,y) location in grid-world
        obs_dim = env_configs['map_height'] * env_configs['map_width']
    else:
        obs_dim = train_env.observation_space.shape[0]
    # Set action specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    print('is_discrete', is_discrete)
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]
    action_low, action_high = None, None
    if isinstance(sampling_env.action_space, gym.spaces.Box):
        action_low, action_high = sampling_env.action_space.low, sampling_env.action_space.high

    # Load expert data
    expert_path = config['running']['expert_path']
    if 'expert_rollouts' in config['running'].keys():
        expert_rollouts = config['running']['expert_rollouts']  # how many rollouts (trajectories) to load
    else:
        expert_rollouts = None
    (expert_obs_rollouts, expert_acs_rollouts, expert_rs_rollouts), expert_mean_reward = load_expert_data(
        expert_path=expert_path,
        num_rollouts=expert_rollouts,
        add_next_step=False,
        log_file=log_file
    )
    #print('expert_obs_rollouts are:',expert_obs_rollouts)
    if config['running']['store_by_game']:
        expert_obs = expert_obs_rollouts
        expert_acs = expert_acs_rollouts
        expert_rs = expert_rs_rollouts
    else:  # concat data for all rollouts
        expert_obs = np.concatenate(expert_obs_rollouts, axis=0)
        expert_acs = np.concatenate(expert_acs_rollouts, axis=0)
        expert_rs = np.concatenate(expert_rs_rollouts, axis=0)

    if 'WGW' in config['env']['train_env_id']:  # visualize the expert trajectories in grid-world env
        traj_visualization_2d(config=config,
                              observations=expert_obs_rollouts,
                              save_path=save_model_mother_dir,
                              model_name=args.config_file.split('/')[-1].split('.')[0],
                              title='Ground-Truth',
                              )

    # add Logger
    if log_file is None:
        icrl_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        icrl_logger = logger.HumanOutputFormat(log_file)

    if "Discrete" in config['task']:
        cn_parameters = {
            'expert_obs': expert_obs,  # select obs at a time step t
            'expert_acs': expert_acs,  # select acs at a time step t
            'device': config['device'],
            'task': config['task'],
            'env_configs': env_configs,
        }
    else:
        # Initialize constraint net, true constraint net
        cn_lr_schedule = lambda x: (config['CN']['anneal_clr_by_factor'] ** (config['running']['n_iters'] * (1 - x))) \
                                   * config['CN']['cn_learning_rate']
        cn_obs_select_name = config['CN']['cn_obs_select_name']
        print(
            "Selecting obs features are : {0}".format(cn_obs_select_name if cn_obs_select_name is not None else 'all'),
            file=log_file, flush=True)
        cn_obs_select_dim = get_input_features_dim(feature_select_names=cn_obs_select_name,
                                                   all_feature_names=None)
        cn_acs_select_name = config['CN']['cn_acs_select_name']
        print(
            "Selecting acs features are : {0}".format(cn_acs_select_name if cn_acs_select_name is not None else 'all'),
            file=log_file, flush=True)
        cn_acs_select_dim = get_input_features_dim(feature_select_names=cn_acs_select_name,
                                                   all_feature_names=None)
        cn_parameters = {
            'obs_dim': obs_dim,
            'acs_dim': acs_dim,
            'hidden_sizes': config['CN']['cn_layers'],
            'batch_size': config['CN']['cn_batch_size'],
            'lr_schedule': cn_lr_schedule,
            'expert_obs': expert_obs,  # select obs at a time step t
            'expert_acs': expert_acs,  # select acs at a time step t
            'is_discrete': is_discrete,
            'regularizer_coeff': config['CN']['cn_reg_coeff'],
            'obs_select_dim': cn_obs_select_dim,
            'acs_select_dim': cn_acs_select_dim,
            'clip_obs': config['CN']['clip_obs'],
            'initial_obs_mean': None if not config['CN']['cn_normalize'] else np.zeros(obs_dim),
            'initial_obs_var': None if not config['CN']['cn_normalize'] else np.ones(obs_dim),
            'action_low': action_low,
            'action_high': action_high,
            'target_kl_old_new': config['CN']['cn_target_kl_old_new'],
            'target_kl_new_old': config['CN']['cn_target_kl_new_old'],
            'train_gail_lambda': config['CN']['train_gail_lambda'],
            'eps': config['CN']['cn_eps'],
            'device': config['device'],
            'task': config['task'],
            'env_configs': env_configs,
            'recon_obs': recon_obs,
        }

    if 'ICRL' == config['group'] or 'Binary' == config['group']:
        if "Discrete" in config['task']:
            constraint_net = ConstraintDiscrete(**cn_parameters)
        else:
            cn_parameters.update({'no_importance_sampling': config['CN']['no_importance_sampling'], })
            cn_parameters.update({'per_step_importance_sampling': config['CN']['per_step_importance_sampling'], })
            constraint_net = ConstraintNet(**cn_parameters)
    elif 'VICRL' == config['group']:
        cn_parameters.update({'di_prior': config['CN']['di_prior'], })
        cn_parameters.update({'mode': config['CN']['mode'], })
        if cn_parameters['mode'] == 'CVaR' or cn_parameters['mode'] == 'VaR':
            cn_parameters['confidence'] = config['CN']['confidence']
        constraint_net = VariationalConstraintNet(**cn_parameters)
    else:
        raise ValueError("Unknown group: {0}".format(config['group']))

    # Pass updated cost_function to cost wrapper (train_env, eval_env, sampling_env)
    train_env.set_cost_function(constraint_net.cost_function)
    sampling_env.set_cost_function(constraint_net.cost_function)

    # visualize the cost function for gridworld
    if 'WGW' in config['env']['train_env_id']:
        ture_cost_function = get_true_cost_function(env_id=config['env']['train_env_id'], env_configs=env_configs)
        #print('true_cost_function', ture_cost_function)
        #input('true_cost_function')
        constraint_visualization_2d(cost_function=ture_cost_function,
                                    feature_range=config['env']["visualize_info_ranges"],
                                    select_dims=config['env']["record_info_input_dims"],
                                    num_points_per_feature=env_configs['map_height'],
                                    obs_dim=train_env.observation_space.shape[0],
                                    acs_dim=1 if is_discrete else train_env.action_space.shape[0],
                                    save_path=save_model_mother_dir,
                                    model_name=args.config_file.split('/')[-1].split('.')[0],
                                    title='Ground-Truth',
                                    )

    # Init agent
    if 'PPO' in config.keys():
        ppo_parameters = load_ppo_config(config=config,
                                         train_env=train_env,
                                         seed=seed,
                                         log_file=log_file)
        create_nominal_agent = lambda: PPOLagrangian(**ppo_parameters)
        reset_policy = config['PPO']['reset_policy']
        reset_every = config['PPO']['reset_every'] if reset_policy else None
        forward_timesteps = config['PPO']['forward_timesteps']
        warmup_timesteps = config['PPO']['warmup_timesteps']
    elif 'iteration' in config.keys():
        iteration_parameters = load_policy_iteration_config(config=config,
                                                            env_configs=env_configs,
                                                            train_env=train_env,
                                                            seed=seed,
                                                            log_file=log_file)
        create_nominal_agent = lambda: PolicyIterationLagrange(**iteration_parameters)
        reset_policy = config['iteration']['reset_policy']
        reset_every = config['iteration']['reset_every']
        forward_timesteps = config['iteration']['max_iter']
        warmup_timesteps = config['iteration']['warmup_timesteps']
    else:
        raise ValueError("Unknown model {0}.".format(config['group']))

    
    # Callbacks
    all_callbacks = []

    # Warmup
    timesteps = 0.
    if warmup_timesteps:
        print("\nWarming up", file=log_file, flush=True)
        input('Warming up')
        with ProgressBarManager(warmup_timesteps) as callback:
            nominal_agent.learn(total_timesteps=warmup_timesteps,
                                cost_function=null_cost,  # During warmup we dont want to incur any cost
                                callback=callback)
            timesteps += nominal_agent.num_timesteps
    # monitor the memory and running time
    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Setting model', log_file=log_file)

    # obtain expert policy under true constraint function



    # Train
    start_time = time.time()
    print("\nBeginning training", file=log_file, flush=True)
    best_true_reward, best_true_cost, best_forward_kl, best_reverse_kl = -np.inf, np.inf, np.inf, np.inf
    vareps = 0.1 # target accuracy
    gamma = config['iteration']['gamma']
    vareps_itr = 1/(1-config['iteration']['gamma'])
    vareps_itr_list = []
    itra = 0
    nominal_agent = create_nominal_agent()
    nominal_agent0 = create_nominal_agent()#learn without constraint
    nominal_agent1 = create_nominal_agent()#learn expert policy
    nominal_agent2 = create_nominal_agent()#learn \hat{c_k}
    nominal_agent3 = create_nominal_agent()#learn V^{\hat{\pi^\expert}}
    num_of_active = 50 # number of active sampling per iteration
    lambda_1 = 0 #0 for 1st constraint                                                                                                            
    lambda_2 = 0 #0 for 1st constraint 
    eps = 0
    constant = 0.3#0.3 for 1st constraint
    kappa = 0.1
    x = env_active.get_initial_occupancy_measure()
    GIoU = []
    plot_itra = [10, 20, 25, 30, 35, 40, 45, 50, 100, 200, 500]
    rewards_expert = []
    costs_expert = []
    rewards_agent = []
    costs_agent = []
    num_of_itra = 500
    #expert_value_function1 = 1/(1-config['iteration']['gamma'])*np.array()
    #cost_k = np.zeros((height=env_configs['map_height'], width=env_configs['map_width'], n_actions=env_configs['n_actions']))



    while vareps_itr > vareps:

	# active sampling, update for estimated transition and estimated expert policy
        if itra == 0:
            estimated_transition, sample_count, expert_policy_active = env_active.active_sampling(0)
        else:
            estimated_transition, sample_count, expert_policy_active = env_active.active_sampling(num_of_active, obs, acs, expert_policy)
        transition = env_active.get_original_transition()
        #print('Active sampling with {} per iteration'.format(num_of_active))#
        #print('transition', np.round(transition,4))
        #input('transition')
        #print('sample_count', np.round(sample_count,1))
        #input('sample_count')

        if itra > num_of_itra: # config['running']['n_iters']:
            break
        else:
            itra += 1
        print('itra:', itra)
        #input('itra')

        if itra == 1:
            # get expert policy for unsafe states, use true cost function
            print('\n#####get expert policy for unsafe states#####\n')
            with ProgressBarManager(forward_timesteps) as callback:
                expert_value_function_unsafe, _ = nominal_agent0.learn(
                total_timesteps=forward_timesteps,
                cost_function=constraint_net.cost_function_zero,  # without constraint
                transition = transition,
                callback=[callback] + all_callbacks
            )
                forward_metrics = logger.Logger.CURRENT.name_to_value
                timesteps += nominal_agent0.num_timesteps
            optimal_policy_without_constraint = nominal_agent0.get_policy()
            expert_policy_unsafe = deepcopy(optimal_policy_without_constraint)#save the value instead of the function
            #print('expert policy for unsafe states:\n',np.round(expert_policy_unsafe,2))
            #input('expert policy unsafe')
            #print('expert value function\n', np.round(expert_value_function_unsafe,3))
            #input('itr:0')

            # get expert policy for safe states, use true cost function
            print('\n#####get expert policy for safe states#####\n')
            with ProgressBarManager(forward_timesteps) as callback:
                expert_value_function, traj_expert = nominal_agent1.learn(
                total_timesteps=forward_timesteps,
                cost_function=ture_cost_function,  # with true constraint
                transition = transition,
                callback=[callback] + all_callbacks
            )
                forward_metrics = logger.Logger.CURRENT.name_to_value
                timesteps += nominal_agent1.num_timesteps
            optimal_policy_with_true_constraint = nominal_agent1.get_policy()
            #print('expert policy for safe states:\n',np.round(optimal_policy_with_true_constraint,2))
            #input('expert_policy_safe')
            expert_policy = deepcopy(optimal_policy_with_true_constraint)
            if env_configs['unsafe_states'] != [[2,0], [2,1], [2,2], [2,3], [2,4], [4,2], [4,3], [4,4], [4,5], [4,6]]:
                for unsafe_state in env_configs['unsafe_states']:
                    #print('unsafe_state', unsafe_state)
                    expert_policy[unsafe_state[0]][unsafe_state[1]] = expert_policy_unsafe[unsafe_state[0]][unsafe_state[1]]
            for cnt in range(num_of_itra):
                rewards_expert.append(cal_discounted_cumulative_rewards(traj=traj_expert, reward_states=env_configs['reward_states'], gamma=gamma))
                costs_expert.append(cal_discounted_cumulative_costs(traj=traj_expert, unsafe_states=env_configs['unsafe_states'], gamma=gamma))
            #print('rewards_expert', rewards_expert)
            #input('rewards_expert')
            #print('expert policy for safe states\n', np.round(expert_policy,2))
            #input('expert policy safe')
            #print('expert policy for complete\n', np.round(expert_policy,2))
            #input('expert policy complete')
            #print('expert value function safe\n', np.round(expert_value_function,3))
            #input('expert value function safe')
            #input('itr:1')

        if itra >= 2: 
            # update \hat{c_k}
            print('\n#####Update c_k#####\n')
    
            with ProgressBarManager(forward_timesteps) as callback:
                expert_value_function, Q_value_function, advantage_function = nominal_agent2.expert_learn(
                total_timesteps=forward_timesteps,
                cost_function=ture_cost_function,  # we do not use this
                expert_policy = expert_policy_active,
                unsafe_states = env_configs['unsafe_states'],
                transition=estimated_transition,
                callback=[callback] + all_callbacks
            )
                forward_metrics = logger.Logger.CURRENT.name_to_value
                timesteps += nominal_agent2.num_timesteps

            #print('expert value function complete\n', np.round(expert_value_function,3),'expert_policy_active', expert_policy_active)
            #input('expert value function complete')
            # update c_k
            constraint_net.train_traj_nn(nominal_obs=[], advantage_function=advantage_function)
            if itra in plot_itra:
                constraint_visualization_2d(cost_function=constraint_net.cost_function,
                                            feature_range=config['env']["visualize_info_ranges"],
                                            select_dims=config['env']["record_info_input_dims"],
                                            num_points_per_feature=env_configs['map_height'],#本来没有这个kw,就会有偏差
                                            obs_dim=train_env.observation_space.shape[0],
                                            acs_dim=1 if is_discrete else train_env.action_space.shape[0],
                                            save_path=save_model_mother_dir+ '/cost_matrix_estimated',
                                            model_name=args.config_file.split('/')[-1].split('.')[0],
                                            title='Iteration-{0}'.format(itra),
                                            force_mode='mean',
                                            )
            GIoU.append(np.round(cal_GIoU(constraint_net.true_cost_matrix,constraint_net.cost_matrix),3))
            #print('constraint_net.true_cost_matrix', constraint_net.true_cost_matrix)
            #print('constraint_net.cost_matrix', constraint_net.cost_matrix)
            print('GIoU', itra, cal_GIoU(constraint_net.true_cost_matrix,constraint_net.cost_matrix))
            #input('constraint_net.true_cost_matrix')
            
            #print('Q value function complete\n', np.round(Q_value_function,3))
            #print('advantage function complete\n', np.round(advantage_function,3))
            #print('sample_count', sample_count)
            #input('itr:2')
            #print('\n#####learn identified constraint for discounted cumulative rewards and costs#####', itra, '\n')
            for cnt in range(1):
                rewards_tmp = []
                costs_tmp = []
                nominal_agent = create_nominal_agent()
                with ProgressBarManager(forward_timesteps) as callback:                
                    _, traj =nominal_agent.learn(
                    total_timesteps=forward_timesteps,
                    cost_function=constraint_net.cost_function,  # Cost should come from cost wrapper
                    transition=estimated_transition,
                    callback=[callback] + all_callbacks
                )
                    forward_metrics = logger.Logger.CURRENT.name_to_value
                    timesteps += nominal_agent.num_timesteps                
                rewards_tmp.append(np.round(cal_discounted_cumulative_rewards(traj=traj, reward_states=env_configs['reward_states'], gamma=gamma),5))
                costs_tmp.append(np.round(cal_discounted_cumulative_costs(traj=traj, unsafe_states=env_configs['unsafe_states'], gamma=gamma),5))
            rewards_tmp=np.array(rewards_tmp)
            rewards_agent.append(np.max(rewards_tmp[np.where(costs_tmp==np.min(np.array(costs_tmp)))]))
            costs_agent.append(np.min(np.array(costs_tmp)))


            print("####Learn V^{r,\hat{\pi^\expert}}####")
            with ProgressBarManager(forward_timesteps) as callback:
                expert_value_function1, Q_value_function1, advantage_function1 = nominal_agent3.expert_learn(
                total_timesteps=forward_timesteps,
                cost_function=constraint_net.cost_function_zero,  # we do not use this
                expert_policy = expert_policy_active,
                unsafe_states = env_configs['unsafe_states'],
                transition=estimated_transition,
                callback=[callback] + all_callbacks
            )
                forward_metrics = logger.Logger.CURRENT.name_to_value
                timesteps += nominal_agent3.num_timesteps

        ci = get_hoeffding_ci_active(height=env_configs['map_height'], width=env_configs['map_width'], n_actions=env_configs['n_actions'],     sample_count=sample_count, v_m=[], zeta_max=config['iteration']['zeta_max'], gamma=config['iteration']['gamma'], 	epsilon=config['iteration']['epsilon'], delta=0.1)
        ci[np.where(np.isnan(ci))]=0
        #print('ci',np.round(ci,2))
        #input('ci')
        vareps_itr = config['iteration']['gamma']*np.max(ci)/(1-config['iteration']['gamma'])
        print('itra, vareps_itr', itra, vareps_itr)#, np.max(sample_count), sample_count)
        #input('vareps_itr')
        np.set_printoptions(suppress=True)
        vareps_itr_list.append(np.round(vareps_itr,2))

        """Implements the two-timescale stochastic approximation"""
        # get v^{c,*}
        v_c = costValueIteration(height=env_configs['map_height'], width=env_configs['map_width'], ci=constraint_net.cost_matrix_sa, n_actions=env_configs['n_actions'], gamma=config['iteration']['gamma'], transition=transition, env=env_active, stopping_threshold=config['iteration']['stopping_threshold'])[env_configs['start_states'][0][0]][env_configs['start_states'][0][0]]
        v_r = expert_value_function1[0][0] if itra >= 2 else 0
        R_k = cal_R_k(gamma, transition, estimated_transition, expert_policy, expert_policy_active, R_max = 1)
        a_k = constant/itra
        b_k = constant/itra**(0.5+kappa)
        gra_of_x = cal_gra_of_x(lambda_1, lambda_2, ci, constraint_net.cost_matrix_sa, env_active.get_reward_mat_sa(), env_active)
        gra_of_lambda_1 = cal_gra_of_lambda_1(gamma, v_c, vareps_itr, eps, x, constraint_net.cost_matrix_sa)
        gra_of_lambda_2 = cal_gra_of_lambda_2(gamma, v_r, R_k, x, env_active.get_reward_mat_sa())  
        #print('v_r',v_r, 'R_k', R_k, 'gra_of_lambda_2', gra_of_lambda_2) 
        #input('parameter of lambda_2')     
        #print('cost matrix\n', constraint_net.cost_matrix_sa, 'reward_sa\n', env_active.get_reward_mat_sa())
        #print('before update\n','occupancy measure\n', np.round(x, 6), 'lambda_1', lambda_1, 'lambda_2', lambda_2)
        #print('x',x,'gra_of_x',gra_of_x)
        #input('before update')
        x = update_x(x, gra_of_x, a_k)                                                                                                           
        lambda_1 = update_lambda_1(lambda_1, gra_of_lambda_1, b_k)
        lambda_2 = update_lambda_2(lambda_2, gra_of_lambda_2, b_k) 
        #print('after update\n','occupancy measure\n', np.round(x, 6), 'lambda_1', lambda_1, 'lambda_2', lambda_2)
        #print('x',x)
        #input('after update')
        pi_expl = cal_pi_expl(height=env_configs['map_height'], width=env_configs['map_width'], n_actions=env_configs['n_actions'], x_k=x, env=env_active, k=itra)
        #print('exploration policy\n', np.round(pi_expl,3),x)
        #input('pi_expl')
        obs, acs = env_active.step_from_pi_expl_active(pi_expl,num_of_active=num_of_active)
        #print('obs, acs', obs, acs, len(obs)) 
        #input('obs, acs')
    #print('ci',np.round(ci,2))
    #input('ci')
    #print('exploration policy', pi_expl)
    #input('pi_expl')


    print('rewards and costs:', rewards_expert[0], costs_expert[0], rewards_agent, costs_agent)
    input('discounted and cumulative rewards and costs')
    #print(sample_count)
    print(GIoU)
    print(vareps_itr_list)
    input('GIoU, vareps_itr')

    for itr in range(3):#range(config['running']['n_iters']):
        if reset_policy and itr % reset_every == 0:
            print("\nResetting agent", file=log_file, flush=True)
            nominal_agent = create_nominal_agent()#learn with identified constraint
        current_progress_remaining = 1 - float(itr) / float(config['running']['n_iters'])
        
        # learn identified constraint
        if itr >= 1:
            print('\n#####learn identified constraint#####', itr, '\n')
            nominal_agent = create_nominal_agent()
            with ProgressBarManager(forward_timesteps) as callback:
                _, traj =nominal_agent.learn(
                total_timesteps=forward_timesteps,
                cost_function=constraint_net.cost_function,  # Cost should come from cost wrapper
                transition=estimated_transition,
                callback=[callback] + all_callbacks
            )
                forward_metrics = logger.Logger.CURRENT.name_to_value
                timesteps += nominal_agent.num_timesteps
        print("v_m", np.round(nominal_agent.get_v_m(),3))
        #print("policy", np.round(nominal_agent.get_policy(),3))
        #input('itr:3')


        # monitor the memory and running time
        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Training PPO model',
                                             log_file=log_file)

        # Sample nominal trajectories, The nominal trajectory is a valid trajectory that is obtained from a human subject
        sync_envs_normalization(train_env, sampling_env)
        orig_observations, observations, actions, rewards, sum_rewards, lengths = sampler.sample_from_agent(
            policy_agent=nominal_agent,
            new_env=sampling_env,
        )



        
        
        if config['running']['use_buffer']:
            sample_data_queue.put(obs=orig_observations,
                                  acs=actions,
                                  rs=rewards,
                                  ls=lengths,
                                  )
            sample_obs, sample_acts, sample_rs, sample_ls = \
                sample_data_queue.get(sample_num=config['running']['sample_rollouts'], )
        else:
            if not config['running']['store_by_game']:
                sample_obs = np.concatenate(orig_observations, axis=0)
                sample_acts = np.concatenate(actions)
                sample_rs = np.concatenate(rewards)
                sample_ls = np.array(lengths)
            else:
                sample_obs = orig_observations
                sample_acts = actions
                sample_rs = rewards
                sample_ls = lengths

        # monitor the memory and running time
        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Sampling',
                                             log_file=log_file)

        # update after we have V(s)
        # Update constraint net
        mean, var = None, None
        if config['CN']['cn_normalize']:
            mean, var = sampling_env.obs_rms.mean, sampling_env.obs_rms.var
        if 'WGW' in config['env']['train_env_id']:  # traj oriented update
            backward_metrics = constraint_net.train_traj_nn(iterations=config['CN']['backward_iters'],
                                                        nominal_obs=sample_obs,
                                                        nominal_acs=sample_acts,
                                                        episode_lengths=sample_ls,
                                                        obs_mean=mean,
                                                        obs_var=var,
                                                        env_configs=env_configs,
                                                        current_progress_remaining=current_progress_remaining,
                                                        advantage_function=advantage_function)
        else:  # normal update
            backward_metrics = constraint_net.train_nn(iterations=config['CN']['backward_iters'],
                                                       nominal_obs=sample_obs,
                                                       nominal_acs=sample_acts,
                                                       episode_lengths=sample_ls,
                                                       obs_mean=mean,
                                                       obs_var=var,
                                                       current_progress_remaining=current_progress_remaining)

        mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                             process_name='Training CN model', log_file=log_file)

        # Pass updated cost_function to cost wrapper (train_env, eval_env, sampling_env)
        train_env.set_cost_function(constraint_net.cost_function)
        sampling_env.set_cost_function(constraint_net.cost_function)

        # Evaluate:
        # reward on true environment
        sync_envs_normalization(train_env, sampling_env)
        save_path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
        if itr % config['running']['save_every'] == 0:
            del_and_make(save_path)
        else:
            save_path = None

        # visualize the trajectories for gridworld
        if 'WGW' in config['env']['train_env_id'] and itr % config['running']['save_every'] == 0:
            traj_visualization_2d(config=config,
                                  observations=orig_observations,
                                  save_path=save_path,
                                  model_name=args.config_file.split('/')[-1].split('.')[0],
                                  title='Iteration-{0}'.format(itr),
                                  )
        # run testing
        mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, costs = \
            evaluate_icrl_policy(nominal_agent, sampling_env,
                                 render=False,
                                 record_info_names=config['env']["record_info_names"],
                                 n_eval_episodes=config['running']['n_eval_episodes'],
                                 deterministic=False,
                                 cost_info_str=config['env']['cost_info_str'],
                                 save_path=save_path, )
        # monitor the memory and running time
        mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                             process_name='Evaluation', log_file=log_file)

        # Save
        # (1) periodically
        if itr % config['running']['save_every'] == 0:
            nominal_agent.save(os.path.join(save_path, "nominal_agent"))
            constraint_net.save(os.path.join(save_path, "constraint_net"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_path, "train_env_stats.pkl"))

            # visualize the cost function
            if 'WGW' in config['env']['train_env_id']:
                constraint_visualization_2d(cost_function=constraint_net.cost_function,
                                            feature_range=config['env']["visualize_info_ranges"],
                                            select_dims=config['env']["record_info_input_dims"],
                                            num_points_per_feature=env_configs['map_height'],#本来没有这个kw,就会有偏差
                                            obs_dim=train_env.observation_space.shape[0],
                                            acs_dim=1 if is_discrete else train_env.action_space.shape[0],
                                            save_path=save_path,
                                            model_name=args.config_file.split('/')[-1].split('.')[0],
                                            title='Iteration-{0}'.format(itr),
                                            force_mode='mean',
                                            )

            else:
                for record_info_idx in range(len(config['env']["record_info_names"])):
                    record_info_name = config['env']["record_info_names"][record_info_idx]
                    plot_record_infos, plot_costs = zip(*sorted(zip(record_infos[record_info_name], costs)))
                    if len(expert_acs.shape) == 1:
                        empirical_input_means = np.concatenate([expert_obs,
                                                                np.expand_dims(expert_acs, 1)],
                                                               axis=1).mean(0)
                    else:
                        empirical_input_means = np.concatenate([expert_obs, expert_acs], axis=1).mean(0)
                    constraint_visualization_1d(cost_function=constraint_net.cost_function,
                                                feature_range=config['env']["visualize_info_ranges"][record_info_idx],
                                                select_dim=config['env']["record_info_input_dims"][record_info_idx],
                                                obs_dim=train_env.observation_space.shape[0],
                                                acs_dim=1 if is_discrete else train_env.action_space.shape[0],
                                                device=constraint_net.device,
                                                save_name=os.path.join(save_path,
                                                                       "{0}_visual.png".format(record_info_name)),
                                                feature_data=plot_record_infos,
                                                feature_cost=plot_costs,
                                                feature_name=record_info_name,
                                                empirical_input_means=empirical_input_means)

        # (2) best
        if mean_nc_reward > best_true_reward:
            # print(utils.colorize("Saving new best model", color="green", bold=True), flush=True)
            print("Saving new best model", file=log_file, flush=True)
            nominal_agent.save(os.path.join(save_model_mother_dir, "best_nominal_model"))
            constraint_net.save(os.path.join(save_model_mother_dir, "best_constraint_net_model"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_model_mother_dir, "train_env_stats.pkl"))

        # Update best metrics
        if mean_nc_reward > best_true_reward:
            best_true_reward = mean_nc_reward

        # Collect metrics
        metrics = {
            "time(m)": (time.time() - start_time) / 60,
            "run_iter": itr,
            "timesteps": timesteps,
            "true/mean_nc_reward": mean_nc_reward,
            "true/std_nc_reward": std_nc_reward,
            "true/mean_reward": mean_reward,
            "true/std_reward": std_reward,
            "best_true/best_reward": best_true_reward
        }

        metrics.update({k.replace("train/", "forward/"): v for k, v in forward_metrics.items()})
        
        # not update the constraint net when itr = 0,1,2
        if itr >= 3:
            metrics.update(backward_metrics)

        # Log
        if config['verbose'] > 0:
            icrl_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)


if __name__ == "__main__":
    args = read_args()
    #print('args are:', args)
    train(args)
