from env.gridworld.multigrid import *
import functools
import matplotlib.pyplot as plt


PLAYER_STR_FORMAT = 'player_{index}'
MAX_CYCLES = 1000

CUSTOM_ID_TO_COLOR_NORMALIZED = {
    1: (0.0, 0.0, 0.0),  # normal empty cell to black
    10: (0.0, 1.0, 1.0),  # agent to cyan
    6: (1.0, 0.0, 0.0),  # ball to red
    2: (0, 0, 1.0), # walls to blue
    0: (1.0, 1.0, 1.0), # unseen to white
}
CUSTOM_ID_TO_COLOR = {
    1: (0, 0, 0),  # normal empty cell to black
    10: (0, 255, 0),  # agent to green
    6: (255, 0, 0),  # ball to red
    2: (0, 0, 255), # walls to blue
    0: (255, 255, 255), # unseen to white
}
SELECTED_AGENT_ORIENTATION_TO_COLOR = {
    0: (0, 255, 0),  # facing right
    1: (0, 195, 0),  # facing down
    2: (0, 135, 0),  # facing left
    3: (0, 75, 0),  # facing up
}   
SELECTED_AGENT_ORIENTATION_TO_COLOR_NORMALIZED = {
    0: (0.0, 1.0, 0.0),  # facing right color green
    1: (0.0, 195.0/255.0, 0.0),  # facing down
    2: (0.0, 135.0/255.0, 0.0),  # facing left
    3: (0.0, 75.0/255.0, 0.0),  # facing up
}
OTHER_AGENT_ORIENTATION_TO_COLOR = {
    0: (0, 255, 255),  # facing right
    1: (0, 195, 195),  # facing down
    2: (0, 135, 135),  # facing left
    3: (0, 75, 75),  # facing up
}
OTHER_AGENT_ORIENTATION_TO_COLOR_NORMALIZED = {
    0: (0.0, 1.0, 1.0),  # facing right color cyan
    1: (0.0, 195.0/255.0, 195.0/255.0),  # facing down
    2: (0.0, 135.0/255.0, 135.0/255.0),  # facing left
    3: (0.0, 75.0/255.0, 75.0/255.0),  # facing up
}

class CollectGameActions:
    available=['still', 'left', 'right', 'forward', 'pickup']

    still = 0
    # Turn left, turn right, move forward
    left = 1
    right = 2
    forward = 3

    # Pick up an object
    pickup = 4


class CollectGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_balls=[],
        agents_index = [],
        balls_index=[],
        balls_reward=[],
        increase_obs_size = 0,
        zero_sum = False,
        max_steps = 1000,
        view_size=7,
        reward_type = "shared",
        partial_obs = True,
        add_agent_id = False,
        normalize_obs = True,
    ):
        """
        agent_index and balls_index are used to decide the color of agents and balls
        :args size: size of the grid
        :args num_balls: number of balls to be placed in the grid for each ball type
        :args agents_index: index of the agents used to decide the color of the agents
        :args balls_index: index of the balls used to decide the color of the balls
        :args balls_reward: reward for each collecting a ball type
        :args increase_obs_size: increase the size of the observation by replicating pixels
        :args zero_sum: whether to use zero sum reward or not
        :args max_steps: maximum number of steps before the episode is terminated
        :args view_size: size of the agent's view (this should be odd to properly work)
        :args reward_type: individual or shared. Shared return the mean of the rewards to all the agents
        :args partial_obs: whether to use partial observations or not
        :args add_agent_id: whether to add the agent id to the observation or not 
        :args normalize_obs: whether to normalize the observation between 0 and 1 or not
        """
        self.num_balls = num_balls
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum
        self.increase_obs_size = increase_obs_size
        self.remaining_balls = num_balls[0]
        self.view_size = view_size if partial_obs else size
        self._num_players = len(agents_index)
        self.reward_type = reward_type
        self.partial_obs = partial_obs
        self.add_agent_id = add_agent_id
        self.world = World
        self.bits_for_id = self._num_players.bit_length() # how many bits are needed to represents this number of agents (e.g. 3 agents -> 2 bits)
        self.normalize_obs = normalize_obs

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= max_steps,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size,
            partial_obs=partial_obs,
            actions_set=CollectGameActions
        )
        if self.add_agent_id:
            # self.observation_space = spaces.Box(low=0, high=1, shape=( self.view_size, self.view_size, 3+self.bits_for_id), dtype=np.float32)
            # RGB OBSERVATIONS
            self.observation_space = spaces.Box(low=0, high=1, shape=(3+self.bits_for_id, self.view_size, self.view_size), dtype=np.float32)
        else:
            # self.observation_space = spaces.Box(low=0, high=1, shape=( self.view_size, self.view_size, 3), dtype=np.float32)
            # RGB OBSERVATIONS
            self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.view_size, self.view_size), dtype=np.float32)



    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for number, index, reward in zip(self.num_balls, self.balls_index, self.balls_reward):
            for i in range(number):
                self.place_obj(Ball(self.world, index, reward))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)


    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        for j,a in enumerate(self.agents):
            if j==i:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, self.agents[i].index]:
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self.remaining_balls = self.remaining_balls - 1
                    self._reward(i, rewards, fwd_cell.reward)

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def step(self, actions):
        obs, rewards, terminated, truncated, info = MultiGridEnv.step(self, actions)
        if self.remaining_balls == 0:
            terminated = True
            rewards += [100.0 for _ in self.agents]
            # normalize the reward 
        if truncated:
            terminated = True
            # rewards += [-1.0 for _ in self.agents]
        if self.reward_type == "shared":
            rewards = np.mean(rewards)
        rgb_obs = self.to_rgb(obs)
        # normalize the reward between ~ [-1, 1]
        if self.increase_obs_size != 0:
            rgb_obs = [self.resize_image(o, height_scale=self.increase_obs_size, width_scale=self.increase_obs_size) for o in rgb_obs]
        rewards = rewards
        return rgb_obs, rewards, terminated, info # removed truncated after terminated

    def reset(self):
        obs, _ = super().reset()
        rgb_obs = self.to_rgb(obs)
        if self.increase_obs_size != 0:
            rgb_obs = [self.resize_image(o, height_scale=self.increase_obs_size, width_scale=self.increase_obs_size) for o in rgb_obs]
        self.remaining_balls = self.num_balls[0]
        return rgb_obs


    def get_avail_actions(self):
        avail_actions = [np.ones(self.action_space(agent_id).n, dtype = self.action_space(agent_id).dtype) for agent_id in range(self.num_agents)]
        return avail_actions
    
    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.action_space.n, dtype = self.action_space.dtype)
    
    def get_obs(self):
        """Returns the observation of the agents in the environment
        :return: list of observations of the agents
        """
        if self.partial_obs:
            obs = [np.moveaxis(obs, -1, 0) for obs in self.gen_obs()]
        else:
            obs = [self.grid.encode(self.objects) for _ in range(self._num_players)]
        rgb_obs = self.to_rgb(obs)
        return rgb_obs


    def get_state(self):
        return np.concatenate(self.get_obs(), axis=0).astype(np.float32)


    def to_rgb(self, obs):
        """
        Convert the observation to RGB using the colors defined in CUSTOM_ID_TO_COLOR
        :param obs: all agents obs
        obs is
        """
        # TODO: problem here self.view_size is 10 but the actual shape of the obs should be 10 
        if obs[0].shape != (6, self.view_size,self.view_size):
            new_obs = [np.moveaxis(o, -1, 0) for o in obs]
        else: 
            new_obs = obs


        rgb_obs = np.zeros([len(self.agents), 3, self.view_size, self.view_size])
        agent_pos_encoding = np.zeros([len(self.agents), 3, self.view_size, self.view_size])
        if self.add_agent_id:
            rgb_obs_plus_id = np.zeros([len(self.agents), 3+self.bits_for_id, self.view_size, self.view_size])
        for a_ind, o in enumerate(new_obs):
            for i, r in enumerate(o[0]):
                for j, c in enumerate(r):
                    # if the cell is an agent cell, then color the agent based on its orientation
                    if not self.partial_obs:
                        if c == 10 and all(self.agents[a_ind].pos == np.array([i,j])):
                            rgb_obs[a_ind][:,i,j] = SELECTED_AGENT_ORIENTATION_TO_COLOR_NORMALIZED[self.agents[a_ind].dir] if self.normalize_obs else SELECTED_AGENT_ORIENTATION_TO_COLOR[self.agents[a_ind].dir]
                        elif c == 10:
                            rgb_obs[a_ind][:,i,j] = OTHER_AGENT_ORIENTATION_TO_COLOR_NORMALIZED[self.grid.get(i,j).dir] if self.normalize_obs else OTHER_AGENT_ORIENTATION_TO_COLOR[self.grid.get(i,j).dir]
                        else:
                            rgb_obs[a_ind][:,i,j] = CUSTOM_ID_TO_COLOR[c] if not self.normalize_obs else CUSTOM_ID_TO_COLOR_NORMALIZED[c]
                    else:
                        if c == 10 and not (i == self.view_size//2 and j == self.view_size-1):
                            # take the relative orientation 
                            orientation_viewer = o[4][self.view_size//2, self.view_size-1]
                            orientation_other_agent = o[4][i,j]
                            # orientation starts from right 0 and goes clockwise
                            relative_orientation = relative_direction(orientation_viewer, orientation_other_agent)
                            rgb_obs[a_ind][:,i,j] = OTHER_AGENT_ORIENTATION_TO_COLOR_NORMALIZED[relative_orientation] if self.normalize_obs else OTHER_AGENT_ORIENTATION_TO_COLOR[relative_orientation]
                        else:
                            rgb_obs[a_ind][:,i,j] = CUSTOM_ID_TO_COLOR[c] if not self.normalize_obs else CUSTOM_ID_TO_COLOR_NORMALIZED[c]

            if self.add_agent_id:
                rgb_obs_plus_id[a_ind] = np.concatenate([rgb_obs[a_ind], self.compute_agent_id_pattern(a_ind)], axis=0)
        if self.add_agent_id:
            return rgb_obs_plus_id
        return rgb_obs
    
    def resize_image(self, image, height_scale=5, width_scale=5):
        """
        Resize the input RGB image by replicating pixels.
        Parameters:
            image (numpy.ndarray): Input RGB image with shape (3, height, width).
            height_scale (int): Scaling factor for the height.
            width_scale (int): Scaling factor for the width.
            
        Returns:
            numpy.ndarray: Resized image with shape (3, height * height_scale, width * width_scale).
        """

        # Get the dimensions of the original image
        channels, height, width = image.shape
        
        # Resize the image by replicating pixels
        resized_height = height * height_scale
        resized_width = width * width_scale
        
        # Create an empty array for the resized image
        resized_image = np.empty((channels, resized_height, resized_width), dtype=np.float32)
        
        # Replicate pixels in the height and width directions
        for i in range(resized_height):
            for j in range(resized_width):
                original_row = i // height_scale
                original_col = j // width_scale
                resized_image[:, i, j] = image[:, original_row, original_col]
        
        return resized_image
    
    def render_obs(self, obs, tile_size=32):
        """
        Render an agent observation for visualization.
        Parameters:
            obs (numpy.ndarray): RGB array of shape (3, height, width).
            tile_size (int): Size of the image tile in pixels.
        """
        # Render the image
        if obs.shape[0] == 3:
            img = np.moveaxis(obs, 0, -1)
        else:
            img = obs
        plt.imshow(img)
        plt.show()


    def compute_agent_id_pattern(self, agent_id):
        """
        Compute the agent id pattern to be added to the observation
        The id pattern is a matrix of size (self.bits_for_id, self.view_size, self.view_size) where each channel
        is a matrix with all zeros or ones representing a bit of the agent id binary representation.
        """
        id_binary_representation = bin(agent_id)[2:]
    
        # Pad the binary representation with leading zeros to the fixed size
        padded_binary = id_binary_representation.zfill(self.bits_for_id)
    
        # Find indexes where there is a '1'
        indexes_of_ones = [index for index, bit in enumerate(padded_binary) if bit == '1']
    
        id_pattern = np.zeros([self.bits_for_id, self.view_size, self.view_size])
        id_pattern[indexes_of_ones, :, :] = 1

        return id_pattern
    
    def unnormalize_obs(self, obs):
        """
        Unnormalize the observation between 0 and 255
        """
        return int(obs*255.0)


class _CollectGamePettingZooEnv(CollectGameEnv):
    def __init__(self, env_config):
        """
        :param env_config: A config dict for the environment.
        :param reward_type: The type of reward to use. Either "shared" or "individual".
        """
        self.env_name = "collect_game"
        self.env_config = env_config
        CollectGameEnv.__init__(self,**env_config)
        self.possible_agents = [
            PLAYER_STR_FORMAT.format(index=index)
            for index in range(self._num_players)
        ]
        # self.individual_observation_names = env_config.individual_observation_names
        observation_space = self.observation_space
        # lru_cache is used to cache the observation and action spaces
        # if the agent_id is the same.
        self.observation_space = functools.lru_cache(
            maxsize=None)(lambda agent_id: observation_space)
        action_space = self.action_space
        self.action_space = functools.lru_cache(maxsize=None)(
            lambda agent_id: action_space)


    def reset(self, seed=None):
        """See base class."""
        obs = super().reset()
        observations = {
            agent: {"RGB": obs[index], "POSITION": self.agents[index].pos, "ORIENTATION": self.agents[index].dir} for index, agent in enumerate(self.possible_agents)
        }
        return observations

    def step(self, action):
        """See base class."""
        if isinstance(action, dict):
            actions = [action[agent] for agent in self.possible_agents]
        else:
            actions = [action[agent] for agent in range(self.num_agents)]
        list_obs, list_rewards, terminated, list_info = super().step(actions)
        observations = {
            agent: {"RGB": list_obs[index], "POSITION": self.agents[index].pos, "ORIENTATION": self.agents[index].dir} for index, agent in enumerate(self.possible_agents)
        }
        rewards = {
            agent: list_rewards[index] for index, agent in enumerate(self.possible_agents)
        }
        # There is the variable self.step_count
        # self.num_cycles += 1
        dones = {agent: terminated for agent in self.possible_agents}

        infos = {agent: {} for agent in self.possible_agents}

        return observations, rewards, dones, infos
    
    def get_avail_actions(self):
        test = [np.ones(self.action_space(agent_id).n, dtype = self.action_space(agent_id).dtype) for agent_id in range(self.num_agents)]
        return test
    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.action_space(agent_id).n, dtype = self.action_space(agent_id).dtype)

    def name_to_id(self, name):
        idx = int(name.split("_")[1])
        return idx

    def idx_to_name(self, idx):
        name = PLAYER_STR_FORMAT.format(index=idx)
        return name

        

def relative_direction(agent1_direction, agent2_direction):
    # Define a mapping of absolute directions to relative directions
    relative_directions = {
        0: {0: 3, 1: 0, 2: 1, 3: 2},  # Agent 1 pointing right
        1: {0: 1, 1: 3, 2: 0, 3: 1},  # Agent 1 pointing down
        2: {0: 1, 1: 2, 2: 3, 3: 0},  # Agent 1 pointing left
        3: {0: 0, 1: 1, 2: 2, 3: 3}   # Agent 1 pointing up
    }

    # Return the relative direction based on the mapping
    return relative_directions[agent1_direction][agent2_direction]



class CollectGame4HEnv10x10N2(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[5],
        agents_index = [1,2,3],
        balls_index=[0],
        balls_reward=[1],
        zero_sum=True)

