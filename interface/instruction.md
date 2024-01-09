运行GridWorld: python train_icrl.py ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1.yaml
采样，即Sample nominal trajectories，sample_from_agent()->single_thread_sample_with_policy()->self.policy_agent.predict()
创建训练和采样环境（以采样环境为例），make_eval_env()->make_env()->env = gym.make()
找'cost'表示的cost function是cost matrix，nominal_agent.learn()->learn()->policy_evaluation()->bellman_update()->info.get()/self.env.get_original_cost()
uniform sampling在sample_from_agent()中，self.env.reset()设置初始位置时，需要随即采样
1. self.start_states = None #us-code, wall_grid_world的reset()中
2. action = random.choice(n_actions) #us-code, policy_iteration_lag的predict中
3. cnt = 0 #us-code, cns_sampler.py, single_thread_sample_with_policy(self)
4. 在uniform sampling时，需要将env_utils.py中的class MujocoExternalSignalWrapper(gym.Wrapper):def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]: obs, reward, done, info = self.env.step_from_us(action)，修改step为step_from_us

uniform sampling: python train_icrl.py ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1.yaml
greedy ICRL: python train_icrl_greedy.py ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1.yaml
active exploration: python train_icrl_active_explore.py ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1.yaml

lambda_1,lambda_2,constant,kappa
1/3/4-constraint: 0,0,0.3,0.1
2nd-constraint: -1000,1000,0.25,0.15
