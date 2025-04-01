import sys
from copy import deepcopy

import numpy as np
import torch

from agent.memory.DreamerMemory import DreamerMemory
from agent.models.DreamerModel import DreamerModel
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout
from agent.optim.utils import advantage, info_nce_loss
from agent.utils.params import get_parameters
from agent.utils.strategy_utils import generate_trajectory_scatterplot
from environments import Env
from networks.dreamer.action import Actor
from networks.dreamer.critic import AugmentedCritic, Critic
from networks.dreamer.trajectory_synthesizer import TrajectorySynthesizerRNN, TrajectorySynthesizerAtt


def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class DreamerLearner:
    """
    Learner has both the actor and critic since it needs to optimize the policy gradient objective
    The controller has only the actor since it's used for inference.
    """
    def __init__(self, config):
        self.config = config
        self.model = DreamerModel(config).to(config.DEVICE).eval()
        self.actor = Actor(config.ACTOR_FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(
            config.DEVICE)
        self.use_communication = config.USE_COMMUNICATION
        self.use_augmented_critic = config.USE_AUGMENTED_CRITIC
        self.test_every = config.TEST_EVERY
        if self.use_augmented_critic:
            self.critic = AugmentedCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)
        else:
            self.critic = Critic(config.FEAT, config.HIDDEN).to(config.DEVICE)
        self.use_strategy_selector = config.USE_STRATEGY_SELECTOR
        self.use_trajectory_synthesizer = config.USE_TRAJECTORY_SYNTHESIZER
        if self.use_trajectory_synthesizer:
            self.trajectory_synthesizer =  TrajectorySynthesizerRNN(config.ACTION_SIZE, config.DETERMINISTIC, config.STOCHASTIC, config.HORIZON,\
                                                               config.TRAJECTORY_SYNTHESIZER_HIDDEN, config.TRAJECTORY_SYNTHESIZER_LAYERS).to(config.DEVICE)
            initialize_weights(self.trajectory_synthesizer, mode='xavier')
        initialize_weights(self.model, mode='xavier')
        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                           config.DEVICE, config.ENV_TYPE)
        self.entropy = config.ENTROPY
        self.use_strategy_advantage = config.USE_STRATEGY_ADVANTAGE
        self.trajectory_synthesizer_scale = config.TRAJECTORY_SYNTHESIZER_SCALE
        self.use_wandb = config.USE_WANDB
        self.step_count = -1
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2

    def init_optimizers(self):
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)
        if self.use_trajectory_synthesizer:
            self.trajectory_synthesizer_list = [self.trajectory_synthesizer, self.actor]
            self.trajectory_synthesizer_optimizer = torch.optim.Adam(get_parameters(self.trajectory_synthesizer_list), lr=self.config.TRAJECTORY_SYNTHESIZER_LR)

    def params(self):
        if self.use_trajectory_synthesizer:
            return {'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                    'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                    'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()},
                    'trajectory_synthesizer': {k: v.cpu() for k, v in self.trajectory_synthesizer.state_dict().items()}}
        else:
            return {'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                    'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                    'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])
        self.replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                  rollout['fake'], rollout['last'], rollout.get('avail_action'), rollout["neighbors_mask"])
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        for i in range(self.config.MODEL_EPOCHS):
            samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
            self.train_model(samples)

        for i in range(self.config.EPOCHS):
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
            self.train_agent(samples)

    def train_model(self, samples):
        self.model.train()
        loss = model_loss(self.config, self.model, samples['observation'], samples['action'], samples['av_action'],
                          samples['reward'], samples['done'], samples['fake'], samples['last'])
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP)
        self.model.eval()

    def train_agent(self, samples):
        # requirs_grad: actions=False, av_actions=False, old_policy=False, imag_feat=False, returns=False
        actions, av_actions, old_policy, imag_feat, returns = actor_rollout(samples['observation'],
                                                                            samples['action'],
                                                                            samples['last'], self.model,
                                                                            self.actor,
                                                                            self.critic if self.config.ENV_TYPE == Env.STARCRAFT
                                                                            else self.old_critic,
                                                                            self.config,
                                                                            samples['neighbors_mask'])
        if self.use_strategy_advantage:
            strategy_advantage = []
            mean_strategy_value = torch.mean(returns.detach(), dim=0)
            for strat in imag_feat:
                strategy_advantage.append(mean_strategy_value - self.critic(strat).detach())
            adv = torch.stack(strategy_advantage, dim = 0)
        else:
            adv = returns.detach() - self.critic(imag_feat).detach()
        if self.config.ENV_TYPE == Env.STARCRAFT:
            adv = advantage(adv, self.use_strategy_selector)
        if self.use_wandb:
            wandb.log({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[1]) if self.use_strategy_selector else np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                if self.use_strategy_selector:
                    loss = actor_loss(imag_feat[:, idx], actions[:, idx], av_actions[:, idx] if av_actions is not None else None,
                                    old_policy[:, idx], adv[:, idx], self.actor, self.entropy, self.config)
                else:
                    loss = actor_loss(imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                    old_policy[idx], adv[idx], self.actor, self.entropy, self.config)
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                if self.use_strategy_selector:
                    val_loss = value_loss(self.critic, imag_feat[:, idx], returns[:, idx])
                else:
                    val_loss = value_loss(self.critic, imag_feat[idx], returns[idx])
                if self.use_wandb and np.random.randint(20) == 9:
                    wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
        # after updating the policy with the ppo routine, let's update the trajectory synthesizer
        if self.use_trajectory_synthesizer:
            # only old_policy requires grad
            actions, av_actions, old_policy, imag_feat, returns = actor_rollout(samples['observation'],
                                                                    samples['action'],
                                                                    samples['last'], self.model,
                                                                    self.actor,
                                                                    self.critic if self.config.ENV_TYPE == Env.STARCRAFT
                                                                    else self.old_critic,
                                                                    self.config,
                                                                    samples['neighbors_mask'],
                                                                    detach_results=False)
            trajectories = torch.cat([imag_feat, actions], dim=-1)
            traj_embed = []
            for traj in range(len(trajectories)):
                traj_embed.append(self.trajectory_synthesizer(trajectories[traj]))

            traj_embed = torch.stack(traj_embed, dim=0)
            traj_embed_fig = generate_trajectory_scatterplot(traj_embed)
            if self.use_wandb and np.random.randint(100) == 9:
                wandb.log({'Plots/Trajectory_Embeddings': wandb.Image(traj_embed_fig)})
            ts_loss = info_nce_loss(traj_embed) * self.trajectory_synthesizer_scale
            self.apply_optimizer(self.trajectory_synthesizer_optimizer, self.trajectory_synthesizer_list, ts_loss, self.config.GRAD_CLIP)
            if self.use_wandb:
                wandb.log({'Agent/ts_loss': ts_loss.mean()})

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        if type(model) == list:
            torch.nn.utils.clip_grad_norm_(get_parameters(model), grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
