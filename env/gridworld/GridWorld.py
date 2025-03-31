from env.gridworld.collect_game import CollectGameEnv
import torch

class GridWorld:
    def __init__(self, env_name):
        if env_name == "collect_game":
            env_config = {"size": 10, "num_balls" : [5], "agents_index" : [1,1,1], "balls_index":[1], "balls_reward":[10.0],
                        "add_agent_id": False, "partial_obs": False, "normalize_obs": True, "max_steps": 300 ,
                        "view_size":5, "reward_type": "individual", "increase_obs_size": 0,
                      
            }
            self.env = CollectGameEnv(**env_config)
        else: 
            raise Exception("Unknown environment")

        env_info = self.get_env_info()

        self.n_obs = env_info["obs_shape"]
        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]
        self.max_time_steps = self.env.max_steps

    def to_dict(self, l):
        return {i: e for i, e in enumerate(l)}

    def get_env_info(self):
        return {"obs_shape": self.env.observation_space.shape, "n_actions": self.env.action_space.n, "n_agents": self.env._num_players}

    def step(self, action_dict):
        obs, reward, done, info = self.env.step(action_dict)
        # self.to_dict(self.env.get_obs())
        return self.to_dict(obs), {i: reward[i] for i in range(self.n_agents)}, \
               {i: done for i in range(self.n_agents)}, info

    def find_neighbors(self, max_distance=10**10):
        """Find neighbors based on their distances
        :param max_distance: maximum distance between agents
        :return neighbors_mask: torch.Tensor(n_agents, n_agents) tensor of 0s and 1s
        """
        agent_positions = {i: [self.env.agents[i].pos[0], self.env.agents[i].pos[1]]  for i in range(self.n_agents)}
        neighbors_mask = torch.zeros(self.n_agents, self.n_agents)
        for agent_id, agent_pos in agent_positions.items():
            for other_agent_id, other_agent_pos in agent_positions.items():
                if agent_id < other_agent_id:
                    x_dist = agent_pos[0] - other_agent_pos[0]
                    y_dist = agent_pos[1] - other_agent_pos[1]
                    distance = (x_dist**2 + y_dist**2)**0.5  # Euclidean distance
                    if distance <= max_distance:
                        neighbors_mask[agent_id, other_agent_id] = 1.
                        neighbors_mask[other_agent_id, agent_id] = 1.
                elif agent_id == other_agent_id:
                    neighbors_mask[agent_id, agent_id] = 1.
        
        if (neighbors_mask.transpose(0, 1) == neighbors_mask).all():
            pass
        else:
            raise ValueError("Neighbors mask is not symmetric")
        return neighbors_mask

    def reset(self):
        self.env.reset()
        return {i: obs for i, obs in enumerate(self.env.get_obs())}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_avail_agent_actions(self, handle):
        return self.env.get_avail_agent_actions(handle)
