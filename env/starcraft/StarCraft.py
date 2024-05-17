from smac.env import StarCraft2Env
import torch

class StarCraft:

    def __init__(self, env_name):
        self.env = StarCraft2Env(map_name=env_name, continuing_episode=True, difficulty="7")
        env_info = self.env.get_env_info()

        self.n_obs = env_info["obs_shape"]
        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]

    def to_dict(self, l):
        return {i: e for i, e in enumerate(l)}

    def step(self, action_dict):        
        reward, done, info = self.env.step(action_dict)
        
        return self.to_dict(self.env.get_obs()), {i: reward for i in range(self.n_agents)}, \
               {i: done for i in range(self.n_agents)}, info

    def find_neighbors(self, max_distance=10**10):
        """Find neighbors based on their distances
        :param max_distance: maximum distance between agents
        :return neighbors_mask: torch.Tensor(n_agents, n_agents) tensor of 0s and 1s
        """
        agent_positions = {i: [self.env.agents[i].pos.x, self.env.agents[i].pos.y]  for i in range(self.n_agents)}
        neighbors_mask = torch.zeros(self.env.n_agents, self.env.n_agents)
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
