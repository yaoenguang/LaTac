import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from src.envs.mpe.multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, scenario=None, reset_callback=None, reward_callback=None,
                 observation_callback=None, get_states_masks=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        self.scenario = scenario
        self.max_num_agents = scenario.max_num_agents
        self.max_num_landmarks = scenario.max_num_landmarks
        self.current_step = 0
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.get_states_masks = get_states_masks
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # pymarl框架不需要考虑以下这些参数，所有属性包括obs_shape, state_shape, actions_onehot等都在get_env_info()中定义，但是action_shape的定义需要在step函数中用于action的执行
        # 实现：初始化policy network输入输出时选取entitites数量最多的scenario，之后每次reset时再重新选取其他scenario.

        # configure spaces according to the scenario with the maximum number of agents and non-agents
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    # def step(self, action_n):
    #     obs_n = []
    #     reward_n = []
    #     done_n = []
    #     info_n = {'n': []}
    #     self.agents = self.world.policy_agents
    #     # set action for each agent
    #     for i, agent in enumerate(self.agents):
    #         self._set_action(action_n[i], agent, self.action_space[i])
    #     # advance world state
    #     self.world.step()
    #     # record observation for each agent
    #     for agent in self.agents:
    #         obs_n.append(self._get_obs(agent))
    #         reward_n.append(self._get_reward(agent))
    #         done_n.append(self._get_done(agent))
    #
    #         info_n['n'].append(self._get_info(agent))
    #
    #     # all agents get total reward in cooperative case
    #     reward = np.sum(reward_n)
    #     if self.shared_reward:
    #         reward_n = [reward] * self.n
    #
    #     return obs_n, reward_n, done_n, info_n

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # global states为 (max_num_agents+max_num_landmarks) X 6 维度，
        # full_obs_mask维度为 (max_num_agents+max_num_landmarks) X (max_num_agents+max_num_landmarks)维度，可观测为0，不可观测为1
        # scenario_mask维度为 (max_num_agents + max_max_num_landmarks)
        global_states, full_obs_mask, scenario_mask = self._get_states_masks()
        for agent in self.agents:
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            obs_n.append(self._get_obs(agent))

            # info_n['n'].append(self._get_info(agent))
        # The length of obs should be max_num_agents to make the batch learning more convenient.
        # for i in range(self.max_num_agents):
        #     obs_i = global_states * (1 - full_obs_mask[i])
        #     obs_i = np.concatenate(obs_i, axis=0)                       # n_entities * 6
        #     obs_n.append(obs_i)

        # 当前timestep结束，将scenario.step_pass再次置False
        # self.scenario.step_pass = False
        global_states = np.concatenate(obs_n, axis=0)
        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            # reward_n = [reward] * self.n
            reward_n = reward
        else:
            raise Exception("The cooperative environment must return 1-dim reward vector.")
        done_n = np.all(done_n)
        # masks are used to extra the actions of agents located in the observable field.
        masks = (full_obs_mask, scenario_mask)

        self.current_step += 1
        return global_states, obs_n, masks, reward_n, done_n, info_n

    # def reset(self):
    #     # reset world
    #     self.reset_callback(self.world)
    #     # reset renderer
    #     self._reset_render()
    #     # record observations for each agent
    #     obs_n = []
    #     self.agents = self.world.policy_agents
    #     for agent in self.agents:
    #         obs_n.append(self._get_obs(agent))
    #     return obs_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        self.current_step = 0
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        global_states, full_obs_mask, scenario_mask = self._get_states_masks()
        # self.scenario.step_pass = False
        for agent in self.agents:
            obs_i = self._get_obs(agent)
            obs_n.append(obs_i)
        global_states = np.concatenate(obs_n, axis=0)
        # # for i, agent in enumerate(self.agents):
        # for i in range(self.max_num_agents):
        #     obs_i = global_states * (1 - full_obs_mask[i])
        #     obs_i = np.concatenate(obs_i, axis=0)
        #     obs_n.append(obs_i)
        # global_states = np.concatenate(global_states, axis=0)
        masks = (full_obs_mask, scenario_mask)
        cur_task_index = self.scenario.task_index
        return global_states, obs_n, masks, cur_task_index

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            # return {}
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # New adding
    def _get_states_masks(self):
        if self.get_states_masks is None:
            raise Exception("Get_states_masks must be defined.")
        return self.get_states_masks(self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            if (self.current_step + 1) >= self.scenario.episode_limit:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.scenario.max_num_agents):
            if self.discrete_action_space:
                avail_act = [1, 1, 1, 1, 1]
            else:
                raise Exception("The action space must be discrete.")
            avail_actions.append(avail_act)
        return avail_actions

    def get_stats(self):
        return {}

    def get_env_info(self):
        if self.discrete_action_space:
            env_info = {"state_shape": self.scenario.state_size,
                        "obs_shape": self.scenario.local_obs_size,
                        "n_actions":  self.world.dim_p * 2 + 1,
                        "n_agents": self.scenario.max_num_agents,
                        "n_entities": self.scenario.max_num_agents + self.scenario.max_num_landmarks,
                        "episode_limit": self.scenario.episode_limit,
                        "n_tasks": self.scenario.num_tasks}
        else:
            raise Exception("The action space must be discrete.")
        return env_info

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
# class BatchMultiAgentEnv(gym.Env):
#     metadata = {
#         'runtime.vectorized': True,
#         'render.modes' : ['human', 'rgb_array']
#     }
#
#     def __init__(self, env_batch):
#         self.env_batch = env_batch
#
#     @property
#     def n(self):
#         return np.sum([env.n for env in self.env_batch])
#
#     @property
#     def action_space(self):
#         return self.env_batch[0].action_space
#
#     @property
#     def observation_space(self):
#         return self.env_batch[0].observation_space
#
#     def step(self, action_n, time):
#         obs_n = []
#         reward_n = []
#         done_n = []
#         info_n = {'n': []}
#         i = 0
#         for env in self.env_batch:
#             obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
#             i += env.n
#             obs_n += obs
#             # reward = [r / len(self.env_batch) for r in reward]
#             reward_n += reward
#             done_n += done
#         return obs_n, reward_n, done_n, info_n
#
#     def reset(self):
#         obs_n = []
#         for env in self.env_batch:
#             obs_n += env.reset()
#         return obs_n
#
#     # render environment
#     def render(self, mode='human', close=True):
#         results_n = []
#         for env in self.env_batch:
#             results_n += env.render(mode, close)
#         return results_n