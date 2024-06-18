from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch.distributions import OneHotCategorical

from agent.models.DreamerModel import DreamerModel
from networks.dreamer.action import Actor
from networks.dreamer.critic import AugmentedCritic, Critic
from networks.dreamer.rnns import rollout_policy_with_strategies


class DreamerController:
    # learner has both the actor and critic since it needs to optimize the policy gradient objective
    # The controller has only the actor since it's used for inference.
    # In our case we are going to add the critic to the controller since we need it to select the strategy
    def __init__(self, config):
        self.model = DreamerModel(config).eval()
        self.actor = Actor(config.ACTOR_FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS)
        self.use_communication = config.USE_COMMUNICATION
        self.use_augmented_critic = config.USE_AUGMENTED_CRITIC
        self.use_last_state_value = config.USE_LAST_STATE_VALUE
        if self.use_augmented_critic:
            self.critic = AugmentedCritic(config.FEAT, config.HIDDEN)
        else:
            self.critic = Critic(config.FEAT, config.HIDDEN)
        self.expl_decay = config.EXPL_DECAY
        self.expl_noise = config.EXPL_NOISE
        self.expl_min = config.EXPL_MIN
        self.n_strategies = config.N_STRATEGIES
        self.strategy_duration = config.STRATEGY_DURATION
        self.use_strategy_selector = config.USE_STRATEGY_SELECTOR
        self.current_strategies = None
        self.episode_strategy_duration = {strat: 0 for strat in range(self.n_strategies)}
        self.init_rnns()
        self.init_buffer()

    def receive_params(self, params):
        self.model.load_state_dict(params['model'])
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])

    def init_buffer(self):
        self.buffer = defaultdict(list)

    def init_rnns(self):
        self.prev_rnn_state = None
        self.prev_actions = None

    def dispatch_buffer(self):
        total_buffer = {k: np.asarray(v, dtype=np.float32) for k, v in self.buffer.items()}
        last = np.zeros_like(total_buffer['done'])
        last[-1] = 1.0
        total_buffer['last'] = last
        self.init_rnns()
        self.init_buffer()
        return total_buffer

    def update_buffer(self, items):
        for k, v in items.items():
            if v is not None:
                self.buffer[k].append(v.squeeze(0).detach().clone().numpy())

    @torch.no_grad()
    def step(self, observations, avail_actions, neighbors_mask, group_mask, steps_done):
        """"
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, value estimate, and
        next recurrent state.  Moves inputs to device and returns outputs back
        to CPU, for the sampler.  Advances the recurrent state of the agent.
        (no grad)
        """
        # steps done reset at every episode
        if self.use_strategy_selector:
            if steps_done == 1 or steps_done % self.strategy_duration == 0:
                if self.prev_actions is None:
                    self.prev_actions = torch.zeros(observations.size(0), observations.size(1), self.model.action_size,
                                       device=observations.device)
                if self.prev_rnn_state is None:
                    self.prev_rnn_state = self.model.representation.initial_state(self.prev_actions.size(0), observations.size(1),
                                                            device=observations.device)
                # select which strategy to use for each group
                self.current_strategies = self.select_strategies(group_mask, neighbors_mask, self.model, self.actor, self.critic, self.prev_rnn_state)
                self.episode_strategy_duration[self.current_strategies[0].item()] += 1
        state = self.model(observations, self.prev_actions, self.prev_rnn_state, neighbors_mask)
        feats = state.get_features()
        if self.use_strategy_selector:
            # convert strategy index to one hot encoding
            strategy_encoded = encode_strategy(self.current_strategies, self.n_strategies).unsqueeze(0)
            strategy_feats = torch.cat([feats, strategy_encoded], dim=-1)
            action, pi = self.actor(strategy_feats)
        else:
            action, pi = self.actor(feats)
        if avail_actions is not None:
            pi[avail_actions == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample()

        self.advance_rnns(state)
        self.prev_actions = action.clone()
        return action.squeeze(0).clone()

    def select_strategies(self, group_mask, neighbors_mask, model, actor, critic, prev_rnn_state):
        """ 
        This function selects a strategy for each group
        :args groups_mask: torch.Tensor(n_groups, n_agents) n_group is variable 
        :args neighbors_mask: torch.Tensor(n_heads, n_agents, n_agents)
        :args model: DreamerModel model
        :args actor: Actor model
        :args critic: Critic model
        :return strategies: dict(group_name, strategy)
        """
        current_strategies = torch.zeros(neighbors_mask.size(1), dtype=torch.long)
        for group in group_mask:
            # strategies_last_state shape (n_strategies, n_agents, stoch + deter)
            items = rollout_policy_with_strategies(model.transition, neighbors_mask, model.av_action, self.strategy_duration, actor, prev_rnn_state, self.n_strategies)
            #############################################################################################
            # WE CAN TRY BY USING ONLY THE VALUE OF THE LAST STATE OR A VALUE AVERAGE OVER TRAJECTORIES #
            #############################################################################################
            # items["imag_states"] shape (n_strategies, horizon, batch_size, n_agents, stoch + deter)
            if self.use_last_state_value:
                strategies_last_state = items["imag_states"].get_features()[:,-1].squeeze(1) # [n_strategies, n_agents, stoch + deter]
            else:
                strategies_last_state = torch.mean(items["imag_states"].get_features(), dim=1).squeeze(1) # [n_strategies, n_agents, stoch + deter]

            # strategy_distr shape (n_strategies, n_agents, 1). It has the probability of each strategy for each agent on the first dimension
            softmax_mask = neighbors_mask.repeat(self.n_strategies, 1, 1)
            strategies_distr = torch.nn.Softmax(dim=0)(critic(strategies_last_state, softmax_mask)).squeeze(-1)
            # select only the strategy for the agents in the group
            group_strategies_distr = torch.mean(strategies_distr[:, group==1.], dim=1)
            group_strategy = torch.argmax(OneHotCategorical(logits=group_strategies_distr).sample()).item()
            current_strategies[group==1.] = group_strategy
        return current_strategies
    
    def advance_rnns(self, state):
        self.prev_rnn_state = deepcopy(state)

    def exploration(self, action):
        """
        :param action: action to take, shape (1,)
        :return: action of the same shape passed in, augmented with some noise
        """
        for i in range(action.shape[0]):
            if np.random.uniform(0, 1) < self.expl_noise:
                index = torch.randint(0, action.shape[-1], (1, ), device=action.device)
                transformed = torch.zeros(action.shape[-1])
                transformed[index] = 1.
                action[i] = transformed
        self.expl_noise *= self.expl_decay
        self.expl_noise = max(self.expl_noise, self.expl_min)
        return action

def encode_strategy(strategy, n_strategies):
    """return a one hot encoding representation from a strategy index
    :param strategy: torch.Tensor(n_agents)
    :return: torch.Tensor(n_agents, n_strategies-1)
    """

    strategy_encoded = torch.zeros(strategy.size(0), n_strategies-1) if n_strategies > 1 else torch.zeros(strategy.size(0), 1)
    for i in range(strategy.size(0)):
        strategy_encoded[i, strategy[i]-1] = 1
    return strategy_encoded