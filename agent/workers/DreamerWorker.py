from copy import deepcopy

import ray
import random
import torch
import numpy as np
from collections import defaultdict
from agent.utils.strategy_utils import generate_trajectories
from agent.memory.DreamerMemory import DreamerMemory


from environments import Env


class DreamerWorker:
    """
    The worker is the entity that interacts with the environment using the controller, the
    controller receives the observations and executes the action in the environment.
    """
    def __init__(self, idx, env_config, controller_config):
        self.runner_handle = idx
        self.env = env_config.create_env()
        self.controller = controller_config.create_controller()
        self.in_dim = controller_config.IN_DIM
        self.env_type = env_config.ENV_TYPE

    def _check_handle(self, handle):
        if self.env_type == Env.STARCRAFT or self.env_type == Env.GRIDWORLD:
            return self.done[handle] == 0


    def _select_actions(self, state, steps_done, groups, neighbors_mask):
        avail_actions = []
        observations = []
        fakes = []
        nn_mask =  (1. - neighbors_mask).bool() # nn_mask=None

        for handle in range(self.env.n_agents):
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
        # turn all items into float 
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
        if self.env_type == Env.STARCRAFT:
            state = torch.zeros(1, self.in_dim)
        elif self.env_type == Env.GRIDWORLD:
            state = torch.zeros(1, *self.in_dim)
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
            # actions -> still = 0, left = 1, right = 2, forward = 3, pickup = 4
            next_state, reward, done, info = self.env.step([action.argmax() for i, action in enumerate(actions)])
            next_state, reward, done = self._wrap(deepcopy(next_state)), self._wrap(deepcopy(reward)), self._wrap(deepcopy(done))
            neighbors_mask = (1. - neighbors_mask).clone().detach().bool()
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
                             "reward": torch.zeros_like(self.augment(reward)),
                             "fake": torch.ones_like(fakes),
                             "done": torch.ones_like(self.augment(done)),
                             "avail_action": torch.ones_like(actions),
                             "neighbors_mask": torch.zeros(1, self.env.n_agents, self.env.n_agents, dtype=bool)}
                    self.controller.update_buffer(items)
                    self.controller.update_buffer(items)
                break

        if self.env_type == Env.STARCRAFT:
            reward = 1. if 'battle_won' in info and info['battle_won'] else 0.
        if self.controller.use_strategy_selector:
            strategy_duration = {}
            for strat in self.controller.episode_strategy_duration.keys():
                strategy_duration["strategy_" + str(strat)] = self.controller.episode_strategy_duration[strat]/sum(self.controller.episode_strategy_duration.values())
            self.controller.episode_strategy_duration = {strat: 0 for strat in range(self.controller.n_strategies)}
            return self.controller.dispatch_buffer(), {"idx": self.runner_handle,
                                                   "reward": reward,
                                                   "steps_done": steps_done,
                                                    **strategy_duration}
        return self.controller.dispatch_buffer(), {"idx": self.runner_handle,
                                                   "reward": reward,
                                                   "steps_done": steps_done,
                                                    }
                                                   
    
    
    def eval(self, dreamer_params, n_episodes, learner_config=None, return_all=False, return_strategy_plot=False):
        """evaluate the controller over n_episodes
        :param dreamer_params: dict of parameters for the controller
        :param n_episodes: int, number of episodes to evaluate
        :param return_all: bool, if True return the metrics for each episode
        """
        # TODO add a function that save frames for an episode
        self.controller.receive_params(dreamer_params)
        if learner_config is not None:
            dreamer_memory = DreamerMemory(learner_config.CAPACITY, learner_config.SEQ_LENGTH, learner_config.ACTION_SIZE, learner_config.IN_DIM, 2,
                                           learner_config.DEVICE, learner_config.ENV_TYPE)
        win_rate = 0
        list_win_rate = []
        mean_steps = 0
        list_steps_done = []
        if self.controller.use_strategy_selector:
            list_episode_strategy_duration = {"strategy_" + str(strat): [] for strat in self.controller.episode_strategy_duration.keys()}
        # save plot for a single episode:
        random_ep = random.randint(0, n_episodes-1)
        frames = []
        for i in range(n_episodes):
            state = self._wrap(self.env.reset())
            steps_done = 0
            self.done = defaultdict(lambda: False)
            while True:
                steps_done += 1
                neighbors_mask = self.env.find_neighbors()
                groups = self.create_group(neighbors_mask)
                actions, obs, fakes, av_actions = self._select_actions(state, steps_done, groups, neighbors_mask)
                if i == random_ep and self.env.env_configs[0].ENV_TYPE == Env.GRIDWORLD:
                    frames.append(np.moveaxis(self.env.env.env.render(team_strategy=1, mode="rgb_array"),-1, 0)) # add .env for the gridworld
                elif i == random_ep and self.env.env_configs[0].ENV_TYPE == Env.STARCRAFT:
                    frames.append(np.moveaxis(self.env.env.render(team_strategy=1, mode="rgb_array"),-1, 0)) # 
                next_state, reward, done, info = self.env.step([action.argmax() for i, action in enumerate(actions)])
                next_state, reward, done = self._wrap(deepcopy(next_state)), self._wrap(deepcopy(reward)), self._wrap(deepcopy(done))
                neighbors_mask = (1. - neighbors_mask).clone().detach().bool()
                self.done = done
                if return_strategy_plot:
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
                    if return_strategy_plot:
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
                                    "avail_action": torch.ones_like(actions),
                                    "neighbors_mask": torch.zeros(1, self.env.n_agents, self.env.n_agents, dtype=bool)}
                            self.controller.update_buffer(items)
                            self.controller.update_buffer(items)
                    break
            
            if learner_config is not None:
                rollout = self.controller.dispatch_buffer()
                dreamer_memory.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                    rollout['fake'], rollout['last'], rollout.get('avail_action'), rollout["neighbors_mask"])
            
            reward = 1. if 'battle_won' in info and info['battle_won'] else 0.
            list_win_rate.append(reward)
            win_rate += reward
            mean_steps += steps_done
            list_steps_done.append(steps_done)
            if self.controller.use_strategy_selector:
                for strat in self.controller.episode_strategy_duration.keys():
                    list_episode_strategy_duration["strategy_" + str(strat)].append(self.controller.episode_strategy_duration[strat]/sum(self.controller.episode_strategy_duration.values()))
                if return_all:
                    self.controller.episode_strategy_duration = {strat: 0 for strat in self.controller.episode_strategy_duration.keys()}

        ###########################################################
        # GENERATE TRAJECTORIES FROM THE BUFFER FOR STRATEGY PLOT #
        ###########################################################
        if return_strategy_plot and learner_config is not None:
            samples = dreamer_memory.sample(learner_config.MODEL_BATCH_SIZE)
            with torch.no_grad():
                generate_trajectories(samples, self.controller.model, self.controller.actor, self.controller.critic, self.controller.config, self.controller.trajectory_synthesizer, self.controller.config.USE_WANDB)

        win_rate = win_rate / n_episodes
        mean_steps = mean_steps / n_episodes
        if self.controller.use_strategy_selector:
            if return_all:
                strategy_duration = list_episode_strategy_duration
            else:
                strategy_duration = {}
                for strat in self.controller.episode_strategy_duration.keys():
                    strategy_duration["strategy_" + str(strat)] = self.controller.episode_strategy_duration[strat]/sum(self.controller.episode_strategy_duration.values())
                self.controller.episode_strategy_duration = {strat: 0 for strat in range(self.controller.n_strategies)}

            return {"idx": self.runner_handle,
                    "win_rate": win_rate if not return_all else list_win_rate,
                    "mean_steps": mean_steps if not return_all else list_steps_done,
                    "frames": np.array(frames),
                    **strategy_duration}
        else:
            return {"idx": self.runner_handle,
                    "win_rate": win_rate if not return_all else list_win_rate,
                    "mean_steps": mean_steps if not return_all else list_steps_done,
                    "frames": np.array(frames)}

@ray.remote
class RayDreamerWorker(DreamerWorker):
    """
    same as DreamerWorker but for ray
    """
    def __init__(self, idx, env_config, controller_config):
        super().__init__(idx, env_config, controller_config)
