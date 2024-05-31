from copy import deepcopy

import ray
import random
import torch
from flatland.envs.agent_utils import RailAgentStatus
from collections import defaultdict

from environments import Env


@ray.remote
class DreamerWorker:

    def __init__(self, idx, env_config, controller_config):
        self.runner_handle = idx
        self.env = env_config.create_env()
        self.controller = controller_config.create_controller()
        self.in_dim = controller_config.IN_DIM
        self.env_type = env_config.ENV_TYPE

    def _check_handle(self, handle):
        if self.env_type == Env.STARCRAFT:
            return self.done[handle] == 0
        else:
            return self.env.agents[handle].status in (RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART) \
                   and not self.env.obs_builder.deadlock_checker.is_deadlocked(handle)

    def _select_actions(self, state, steps_done, groups, neighbors_mask):
        avail_actions = []
        observations = []
        fakes = []
        if self.env_type == Env.FLATLAND:
            nn_mask = (1. - torch.eye(self.env.n_agents)).bool()
        else:
            nn_mask =  (1. - neighbors_mask).bool() # nn_mask=None

        for handle in range(self.env.n_agents):
            if self.env_type == Env.FLATLAND:
                for opp_handle in self.env.obs_builder.encountered[handle]:
                    if opp_handle != -1:
                        nn_mask[handle, opp_handle] = False
            else:
                avail_actions.append(torch.tensor(self.env.get_avail_agent_actions(handle)))

            if self._check_handle(handle) and handle in state:
                fakes.append(torch.zeros(1, 1))
                observations.append(state[handle].unsqueeze(0))
            elif self.done[handle] == 1:
                fakes.append(torch.ones(1, 1))
                observations.append(self.get_absorbing_state())
            else:
                fakes.append(torch.zeros(1, 1))
                obs = torch.tensor(self.env.obs_builder._get_internal(handle)).float().unsqueeze(0)
                observations.append(obs)

        observations = torch.cat(observations).unsqueeze(0)
        av_action = torch.stack(avail_actions).unsqueeze(0) if len(avail_actions) > 0 else None
        nn_mask = nn_mask.unsqueeze(0).repeat(8, 1, 1) if nn_mask is not None else None
        actions = self.controller.step(observations, av_action, nn_mask, groups, steps_done)
        return actions, observations, torch.cat(fakes).unsqueeze(0), av_action

    def _wrap(self, d):
        for key, value in d.items():
            d[key] = torch.tensor(value).float()
        return d
                    
    def create_group(self, neighbors_mask):
        """create groups by randomizing the agents and adding neighbors to the group
        :param neighbors_mask: torch.Tensor(n_agents, n_agents) is a tensor of 0s and 1s
        :return group_mask: tensor of shape (n_groups, n_agents) where the number of groups is dynamic
        """
        group_mask = []
        group_id = 0
        agents = [i for i in range(self.env.n_agents)]
        random.shuffle(agents)
        agents_already_in_group = []
        for idx in agents:
            if idx not in agents_already_in_group:
                group_mask.append(torch.zeros(self.env.n_agents))
                group_mask[group_id][idx] = 1
                agents_already_in_group.append(idx)
                for neighbor_id, neighbor in enumerate(neighbors_mask[idx]):
                    if neighbor_id != idx and int(neighbor) == 1 and neighbor_id not in agents_already_in_group:
                        group_mask[group_id][neighbor_id] = 1
                        agents_already_in_group.append(neighbor_id)
                group_id += 1
        group_mask = torch.stack(group_mask, dim=0)
        assert group_mask.sum().item() == self.env.n_agents, f"Group mask sum: {group_mask.sum().item()} != {self.env.n_agents}"

        return group_mask

    def get_absorbing_state(self):
        state = torch.zeros(1, self.in_dim)
        return state

    def augment(self, data, inverse=False):
        aug = []
        default = list(data.values())[0].reshape(1, -1)
        for handle in range(self.env.n_agents):
            if handle in data.keys():
                aug.append(data[handle].reshape(1, -1))
            else:
                aug.append(torch.ones_like(default) if inverse else torch.zeros_like(default))
        return torch.cat(aug).unsqueeze(0)

    def _check_termination(self, info, steps_done):
        if self.env_type == Env.STARCRAFT:
            return "episode_limit" not in info
        else:
            return steps_done < self.env.max_time_steps

    def run(self, dreamer_params):
        self.controller.receive_params(dreamer_params)

        state = self._wrap(self.env.reset())
        steps_done = 0
        self.done = defaultdict(lambda: False)
        while True:
            steps_done += 1
            neighbors_mask = self.env.find_neighbors()
            groups = self.create_group(neighbors_mask)
            actions, obs, fakes, av_actions = self._select_actions(state, steps_done, groups, neighbors_mask)
            next_state, reward, done, info = self.env.step([action.argmax() for i, action in enumerate(actions)])
            next_state, reward, done = self._wrap(deepcopy(next_state)), self._wrap(deepcopy(reward)), self._wrap(deepcopy(done))
            neighbors_mask = deepcopy(torch.tensor(1. - neighbors_mask).bool())
            self.done = done
            self.controller.update_buffer({"action": actions,
                                           "observation": obs,
                                           "reward": self.augment(reward),
                                           "done": self.augment(done),
                                           "fake": fakes,
                                           "avail_action": av_actions,
                                           "neighbors_mask": neighbors_mask,
                                           })

            state = next_state
            if all([done[key] == 1 for key in range(self.env.n_agents)]):
                if self._check_termination(info, steps_done):
                    obs = torch.cat([self.get_absorbing_state() for i in range(self.env.n_agents)]).unsqueeze(0)
                    actions = torch.zeros(1, self.env.n_agents, actions.shape[-1])
                    index = torch.randint(0, actions.shape[-1], actions.shape[:-1], device=actions.device)
                    actions.scatter_(2, index.unsqueeze(-1), 1.)
                    items = {"observation": obs,
                             "action": actions,
                             "reward": torch.zeros(1, self.env.n_agents, 1),
                             "fake": torch.ones(1, self.env.n_agents, 1),
                             "done": torch.ones(1, self.env.n_agents, 1),
                             "avail_action": torch.ones_like(actions) if self.env_type == Env.STARCRAFT else None,
                             "neighbors_mask": torch.zeros(1, self.env.n_agents, self.env.n_agents, dtype=bool)}
                    self.controller.update_buffer(items)
                    self.controller.update_buffer(items)
                break

        if self.env_type == Env.FLATLAND:
            reward = sum(
                [1 for agent in self.env.agents if agent.status == RailAgentStatus.DONE_REMOVED]) / self.env.n_agents
        else:
            reward = 1. if 'battle_won' in info and info['battle_won'] else 0.
        return self.controller.dispatch_buffer(), {"idx": self.runner_handle,
                                                   "reward": reward,
                                                   "steps_done": steps_done}
    
    def eval(self, dreamer_params, n_episodes):
        self.controller.receive_params(dreamer_params)
        win_rate = 0
        mean_steps = 0
        for _ in range(n_episodes):
            state = self._wrap(self.env.reset())
            steps_done = 0
            self.done = defaultdict(lambda: False)
            while True:
                steps_done += 1
                neighbors_mask = self.env.find_neighbors()
                groups = self.create_group(neighbors_mask)
                actions, obs, fakes, av_actions = self._select_actions(state, steps_done, groups, neighbors_mask)
                next_state, reward, done, info = self.env.step([action.argmax() for i, action in enumerate(actions)])
                next_state, reward, done = self._wrap(deepcopy(next_state)), self._wrap(deepcopy(reward)), self._wrap(deepcopy(done))
                neighbors_mask = deepcopy(torch.tensor(1. - neighbors_mask).bool())
                self.done = done

                state = next_state
                if all([done[key] == 1 for key in range(self.env.n_agents)]):
                    break

            if self.env_type == Env.FLATLAND:
                reward = sum(
                    [1 for agent in self.env.agents if agent.status == RailAgentStatus.DONE_REMOVED]) / self.env.n_agents
            else:
                reward = 1. if 'battle_won' in info and info['battle_won'] else 0.
            win_rate += reward
            mean_steps += steps_done
        win_rate = win_rate / n_episodes
        mean_steps = mean_steps / n_episodes
        return {"idx": self.runner_handle, "win_rate": win_rate, "mean_steps": mean_steps}