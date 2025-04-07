import numpy as np
import torch
import wandb
import torch.nn.functional as F
import torch.distributions as td

from agent.optim.utils import rec_loss, compute_return, state_divergence_loss, calculate_ppo_loss, \
    batch_multi_agent, log_prob_loss, info_loss, batch_strat_horizon, batch_multi_agent_horizon
from agent.utils.params import FreezeParameters
from networks.dreamer.rnns import rollout_representation, rollout_policy, rollout_policy_with_strategies


def model_loss(config, model, obs, action, av_action, reward, done, fake, last):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]


    embed = model.observation_encoder(obs.reshape(-1, n_agents, *obs.shape[3:]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)                     # [760, 3, 256] to [19, 40, 3, 256]  

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)
    # In this issue https://github.com/jbr-ai-labs/mamba/issues/7 they explain why they use 
    # use as feature the [stoch_t, deter_t+1] instead of [stoch_t, deter_t]
    feat = torch.cat([post.stoch, deters], -1)
    feat_dec = post.get_features() # feat_dec is [stoch_t, deter_t]
    # fakes means samples from agents that are dead and use an absorbing state
    # i_feat is the output of the hidden layer before the last layer of the observation decoder
    if model.observation_decoder.rgb_input:
        reconstruction_loss, i_feat = rec_loss(model.observation_decoder, 
                                                feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                                obs[:-1].reshape(-1, n_agents, *obs.shape[3:]),
                                                1. - fake[:-1].reshape(-1, n_agents, 1))
    else:
        reconstruction_loss, i_feat = rec_loss(model.observation_decoder,
                                                feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                                obs[:-1].reshape(-1, n_agents, obs.shape[-1]),
                                                1. - fake[:-1].reshape(-1, n_agents, 1))
    reward_loss = F.smooth_l1_loss(model.reward_model(feat), reward[1:])
    pcont_loss = log_prob_loss(model.pcont, feat, (1. - done[1:]))
    av_action_loss = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)

    dis_loss = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))
    div = state_divergence_loss(prior, post, config)
    
    # div is the standard KL divergence between the prior and the posterior
    # reward,reconstruction and pcont are the loss for the reward, observation and the discount factor
    # dis_loss is the information loss which is the cross entropy between the action logits and the actions
    # av_action_loss is the loss for the available actions
    model_loss = div + reward_loss + dis_loss + reconstruction_loss + pcont_loss + av_action_loss
    if config.USE_WANDB and np.random.randint(20) == 4:
        with torch.no_grad():
            shape = prior.logits.shape
            prior_logits = torch.reshape(prior.logits, shape = (*shape[:-1], config.N_CATEGORICALS, config.N_CLASSES))
            temp_prior = td.Independent(td.OneHotCategorical(logits=prior_logits), 1)
            post_logits = torch.reshape(post.logits, shape = (*shape[:-1], config.N_CATEGORICALS, config.N_CLASSES))
            temp_post = td.Independent(td.OneHotCategorical(logits=post_logits), 1)

        wandb.log({'Model/reward_loss': reward_loss, 'Model/div': div, 'Model/av_action_loss': av_action_loss,
                   'Model/reconstruction_loss': reconstruction_loss, 'Model/info_loss': dis_loss,
                   'Model/pcont_loss': pcont_loss, 'Model/prior_entropy': temp_prior.entropy().mean(), 'Model/post_entropy': temp_post.entropy().mean()})

    return model_loss


def actor_rollout(obs, action, last, model, actor, critic, config, neighbors_mask=None, detach_results=True):
    """rollout the actor and the critic in the imagination
    :param obs: The observations, shape (time_steps, batch_size, n_agents, obs_size)
    :param action: The actions, shape (time_steps, batch_size, n_agents, action_size)
    :param last: The last state, shape (batch_size, n_agents, feat_size)
    :param model: The model
    :param actor: The actor
    :param critic: The critic
    :param config: The config
    :param neighbors_mask: The neighbors mask, shape (time_steps, batch_size, n_agents, n_agents)
    :param detach_results: Whether to detach the results
    """
    n_agents = obs.shape[2]
    with FreezeParameters([model]):
        if model.observation_encoder.rgb_input:
            embed = model.observation_encoder(obs.reshape(-1, n_agents, *obs.shape[3:]))  # [batch*(seq_len-1), n_agents, channel, width, height]
        else:
            embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))         # [batch*(seq_len-1), n_agents, feat] [760, 3, 256]
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)                     # [batch, seq_len-1, n_agents, feat] [19, 40 , 3, 256]
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(model.representation, obs.shape[0], embed, action,   
                                                prev_state, last)                           # stoch [18, 40, 3, 1024]
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1)) # stoch [720, 3, 1024]
        if config.USE_STRATEGY_SELECTOR:
            nn_mask = neighbors_mask[:-1].reshape((neighbors_mask.shape[0]-1) * neighbors_mask.shape[1], n_agents, -1) #  [720, 3, 3]
            nn_mask = nn_mask.repeat(8,1,1).detach()
            items = rollout_policy_with_strategies(model.transition, nn_mask, model.av_action, config.HORIZON, actor, post, config.N_STRATEGIES)
        else:
            nn_mask = neighbors_mask[:-1].reshape((neighbors_mask.shape[0]-1) * neighbors_mask.shape[1], n_agents, -1) #  [720, 3, 3]
            nn_mask = nn_mask.repeat(8,1,1).detach()
            items = rollout_policy(model.transition, model.av_action, config.HORIZON, actor, post, neighbors_mask=nn_mask)
    imag_feat = items["imag_states"].get_features() # [n_strategies, horizon, seq_len*batch_size, stoch_t, deter_t]
    if config.USE_STRATEGY_SELECTOR:
        imag_rew_feat = torch.cat([items["imag_states"].stoch[:,:-1], items["imag_states"].deter[:, 1:]], -1) # [stoch_t, deter_t+1]
    else:
        imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1) # [stoch_t, deter_t+1]
    returns = critic_rollout(model, critic, imag_feat, imag_rew_feat, items["actions"],
                             items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])), config)

    if config.USE_STRATEGY_SELECTOR:
        if detach_results:
            output = [items["actions"][:, :-1].detach(),
                    items["av_actions"][:, :-1].detach() if items["av_actions"] is not None else None,
                    items["old_policy"][:, :-1].detach(), imag_feat[:, :-1].detach(), returns.detach()]
        else:
            items["new_policy"] = items["actions"] + items["old_policy"] - items["old_policy"].detach()
            output = [items["actions"][:, :-1],
                    items["av_actions"][:, :-1] if items["av_actions"] is not None else None,
                    items["new_policy"][:, :-1], imag_feat[:, :-1], returns]    
            if config.USE_GLOBAL_TRAJECTORY_SYNTHESIZER:
                return [batch_multi_agent_horizon(v) for v in output]
            else:
                return [batch_strat_horizon(v) for v in output]     
    else:
        output = [items["actions"][:-1].detach(),
            items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
            items["old_policy"][:-1].detach(), imag_feat[:-1].detach(), returns.detach()]
    return [batch_multi_agent(v, n_agents, config.USE_STRATEGY_SELECTOR) for v in output]


def critic_rollout(model, critic, states, rew_states, actions, raw_states, config):
    """
    Compute the returns for the critic
    :param model: The model
    :param critic: The critic
    :param states: The states, shape (n_strategies, horizon, seq_len*batch_size, stoch_t, deter_t) if strategy selector is used else (horizon, seq_len*batch_size, stoch_t, deter_t)
    :param rew_states: The states used to calculate the reward, shape (n_strategies, horizon -1, seq_len*batch_size, stoch_t, deter_t+1) if strategy selector is used else (horizon -1, seq_len*batch_size, stoch_t, deter_t+1)
    :param actions: The actions, shape (n_strategies, horizon, seq_len*batch_size, action_size) if strategy selector is used else (horizon, seq_len*batch_size, action_size)
    :param raw_states: The raw RSSM states, shape (n_strategies*horizon*seq_len*batch_size, stoch_t, deter_t) if strategy selector is used else (horizon*seq_len*batch_size, stoch_t, deter_t)
    :param config: The config
    """
    with FreezeParameters([model, critic]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        # reshape the reward and take the mean over the agents
        if config.USE_STRATEGY_SELECTOR:
            imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:, :-1]
        else:
            imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        # They use the attention also inside the critic to get the value of the states
        value = critic(states)
        discount_arr = model.pcont(rew_states).mean
        if config.USE_WANDB:
            wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                    'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                    'Value/Value': value.mean()})
    returns = compute_return(imag_reward, value[:, :-1] if config.USE_STRATEGY_SELECTOR else value[:-1], discount_arr, bootstrap=value[:,-1] if config.USE_STRATEGY_SELECTOR else value[-1], lmbda=config.DISCOUNT_LAMBDA,
                             gamma=config.GAMMA, use_strategy_selector=config.USE_STRATEGY_SELECTOR)
    return returns


def calculate_reward(model, states, mask=None):
    imag_reward = model.reward_model(states)
    if mask is not None:
        imag_reward *= mask
    return imag_reward


def calculate_next_reward(model, actions, states):
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    next_state = model.transition(actions, states)
    imag_rew_feat = torch.cat([states.stoch, next_state.deter], -1)
    return calculate_reward(model, imag_rew_feat)


def actor_loss(imag_states, actions, av_actions, old_policy, advantage, actor, ent_weight, config):
    if config.USE_STRATEGY_SELECTOR:
        imag_states_plus_strategy = []
        for i in range(len(imag_states)):
            encoded_strategy = torch.zeros(*imag_states[i].shape[:2], config.N_STRATEGIES-1, device=imag_states[i].device) if config.N_STRATEGIES > 1 else torch.zeros(*imag_states[i].shape[:2], 1, device=imag_states[i].device)
            if i > 0:
                encoded_strategy[:, :, i-1] = 1
            imag_states_plus_strategy.append(torch.cat([imag_states[i], encoded_strategy], dim=-1))
        imag_states_plus_strategy = torch.stack(imag_states_plus_strategy, dim=0)
        _, new_policy = actor(imag_states_plus_strategy)
        gather_index = 3
    else:
        _, new_policy = actor(imag_states)
        gather_index = 2
    if av_actions is not None:
        new_policy[av_actions == 0] = -1e10
    actions = actions.argmax(-1, keepdim=True)
    rho = (F.log_softmax(new_policy, dim=-1).gather(gather_index, actions) -
           F.log_softmax(old_policy, dim=-1).gather(gather_index, actions)).exp()
    ppo_loss, ent_loss = calculate_ppo_loss(new_policy, rho, advantage)
    if config.USE_WANDB and np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': ent_loss.mean(), 'Policy/Mean action': actions.float().mean()})
    return (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean()


def value_loss(critic, imag_feat, targets):
    value_pred = critic(imag_feat)
    mse_loss = (targets - value_pred) ** 2 / 2.0
    return torch.mean(mse_loss)
