运行GridWorld: python train_icrl.py ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1.yaml
采样，即Sample nominal trajectories，sample_from_agent()->single_thread_sample_with_policy()->self.policy_agent.predict()
创建训练和采样环境（以采样环境为例），make_eval_env()->make_env()->env = gym.make()
找'cost'表示的cost function是cost matrix，nominal_agent.learn()->learn()->policy_evaluation()->bellman_update()->info.get()/self.env.get_original_cost()
uniform sampling在sample_from_agent()中，self.env.reset()设置初始位置时，需要随即采样
1. self.start_states = None #us-code, wall_grid_world的reset()中
2. action = random.choice(n_actions) #us-code, policy_iteration_lag的predict中
3. cnt = 0 #us-code, cns_sampler.py, single_thread_sample_with_policy(self)