import sys
from copy import deepcopy

import numpy as np
import torch
import wandb

from agent.memory.DreamerMemory import DreamerMemory
from agent.models.DreamerModel import DreamerModel
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout
from agent.optim.utils import advantage, info_nce_loss, info_loss_strategy
from agent.utils.params import get_parameters
from agent.utils.strategy_utils import generate_trajectory_scatterplot
from environments import Env
from networks.dreamer.dense import DenseBinaryModel
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
        self.strategy_duration = config.STRATEGY_DURATION
        self.use_trajectory_synthesizer = config.USE_TRAJECTORY_SYNTHESIZER
        self.use_global_trajectory_synthesizer = config.USE_GLOBAL_TRAJECTORY_SYNTHESIZER
        self.trajectory_synthesizer = None
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

    def init_trajectory_synthesizer(self, n_agents=None):
        """Initialize the trajectory synthesizer
        """
        self.trajectory_synthesizer =  TrajectorySynthesizerRNN(self.config.ACTION_SIZE, self.config.DETERMINISTIC, self.config.STOCHASTIC,\
                                                    self.config.STRATEGY_DURATION, self.config.TRAJECTORY_SYNTHESIZER_HIDDEN,\
                                                    self.config.TRAJECTORY_SYNTHESIZER_LAYERS, n_agents=n_agents).to(self.config.DEVICE)
        # for InfoMax over strategies (predict selected strategy from encoded trajectory)
        if self.use_global_trajectory_synthesizer:
            self.strat_features = DenseBinaryModel((self.config.DETERMINISTIC+self.config.STOCHASTIC+self.config.ACTION_SIZE)*n_agents,\
                                                self.config.N_STRATEGIES, self.config.STRAT_LAYERS, self.config.STRAT_HIDDEN).to(self.config.DEVICE)
        else:
            self.strat_features = DenseBinaryModel(self.config.DETERMINISTIC+self.config.STOCHASTIC+self.config.ACTION_SIZE,\
                                                self.config.N_STRATEGIES, self.config.STRAT_LAYERS, self.config.STRAT_HIDDEN).to(self.config.DEVICE)
        initialize_weights(self.trajectory_synthesizer, mode='xavier')
        initialize_weights(self.strat_features, mode='xavier')
        self.init_trajectory_synthesizer_optimizer()


    def init_optimizers(self):
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)

    def init_trajectory_synthesizer_optimizer(self):
        self.trajectory_synthesizer_list = [self.trajectory_synthesizer, self.strat_features, self.actor]
        # overwrite the actor optimizer with the trajectory synthesizer optimizer
        self.actor_optimizer = torch.optim.Adam(get_parameters(self.trajectory_synthesizer_list), lr=self.config.ACTOR_LR, weight_decay=0.00001)

    def params(self):
        if self.use_trajectory_synthesizer and self.trajectory_synthesizer is not None:
            return {'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                    'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                    'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()},
                    'trajectory_synthesizer': {k: v.cpu() for k, v in self.trajectory_synthesizer.state_dict().items()},
                    'strat_features': {k: v.cpu() for k, v in self.strat_features.state_dict().items()}}
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
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE) # sample size = [seq_len, batch_size, n_agents, feat]
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
                                                                            samples['neighbors_mask'],
                                                                            rollout_len=self.config.HORIZON)
        # TODO: strategy_duration and horizon are not the same, pay attention on how PPO uses policy with longer sequences for each strategy
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
            num_of_updates = len(inds) // step + 1
            ts_inds = np.random.permutation(self.config.BATCH_SIZE*(self.config.SEQ_LENGTH-2))
            ts_step = ts_inds.shape[0] // num_of_updates
            ts_i = 0
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                ts_idx = ts_inds[ts_i:ts_i + ts_step]
                ts_i += ts_step
                if self.use_strategy_selector:
                    if self.use_trajectory_synthesizer:
                        _, _, actions_with_grad, ts_imag_feat, _ = actor_rollout(samples['observation'],
                                                                        samples['action'],
                                                                        samples['last'], self.model,
                                                                        self.actor,
                                                                        self.critic if self.config.ENV_TYPE == Env.STARCRAFT
                                                                        else self.old_critic,
                                                                        self.config,
                                                                        samples['neighbors_mask'],
                                                                        detach_results=False,
                                                                        indices=ts_idx,
                                                                        rollout_len=self.strategy_duration)
                        trajectories = torch.cat([ts_imag_feat, actions_with_grad], dim=-1)
                        traj_embed = [] 
                        # imag_feat.size = [n_strategies, horizon-1, (seq_len-2)*batch_size, n_agents*(stoch_t+deter_t)]
                        for traj in range(len(trajectories)):
                            traj_embed.append(self.trajectory_synthesizer(trajectories[traj]))
                        traj_embed = torch.stack(traj_embed, dim=0)
                        ts_loss = info_nce_loss(traj_embed, multiple_positives=False)  
                        strategy_infomax_loss =  info_loss_strategy(traj_embed, self.strat_features)
                        if self.use_wandb and np.random.randint(100) == 9:
                            with torch.no_grad():
                                traj_embed_fig = generate_trajectory_scatterplot(traj_embed, red_type="tsne")
                                wandb.log({'Plots/Trajectory_Embeddings': wandb.Image(traj_embed_fig)})
                    ac_loss = actor_loss(imag_feat[:, idx], actions[:, idx], av_actions[:, idx] if av_actions is not None else None,
                                    old_policy[:, idx], adv[:, idx], self.actor, self.entropy, self.config)
                    if self.use_trajectory_synthesizer:
                        loss = ac_loss + ts_loss + strategy_infomax_loss
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
                    if self.trajectory_synthesizer:
                        wandb.log({'Agent/val_loss': val_loss.item(), 'Agent/loss': loss.item(), 'Agent/actor_loss': ac_loss.item(), 'Agent/infomax_loss': strategy_infomax_loss.item(), 'Agent/ts_loss': ts_loss.item()})
                    else:
                        wandb.log({'Agent/val_loss': val_loss.item(), 'Agent/actor_loss': loss.item()})
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)


    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        if type(model) == list:
            torch.nn.utils.clip_grad_norm_(get_parameters(model), grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
