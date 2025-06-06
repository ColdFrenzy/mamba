import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rec_loss(decoder, z, x, fake):
    x_pred, feat = decoder(z)
    if decoder.rgb_input:
        batch_size = np.prod(list(x.shape[:-3]))
        sample_loss = F.smooth_l1_loss(x_pred, x, reduction='none')
        sample_loss = sample_loss.reshape(sample_loss.shape[0], sample_loss.shape[1], -1).sum(dim=-1).unsqueeze(dim=-1)
        gen_loss1 = (sample_loss * fake).sum() / batch_size
    else:
        batch_size = np.prod(list(x.shape[:-1]))
        gen_loss1 = (F.smooth_l1_loss(x_pred, x, reduction='none') * fake).sum() / batch_size
    
    return gen_loss1, feat


def ppo_loss(A, rho, eps=0.2):
    return -torch.min(rho * A, rho.clamp(1 - eps, 1 + eps) * A)


def mse(model, x, target):
    pred = model(x)
    return ((pred - target) ** 2 / 2).mean()


def entropy_loss(prob, logProb):
    return (prob * logProb).sum(-1)


def advantage(A, use_strategies=False):
    if use_strategies:
        strat_adv = []
        for a in A:
            std = 1e-4 + a.std() if len(a) > 0 else 1
            adv = (a - a.mean()) / std
            adv = adv.detach()
            adv[adv != adv] = 0
            strat_adv.append(adv)
        return torch.stack(strat_adv, dim=0)
    else:
        std = 1e-4 + A.std() if len(A) > 0 else 1
        adv = (A - A.mean()) / std
        adv = adv.detach()
        adv[adv != adv] = 0
    return adv


def calculate_ppo_loss(logits, rho, A):
    prob = F.softmax(logits, dim=-1)
    logProb = F.log_softmax(logits, dim=-1)
    polLoss = ppo_loss(A, rho)
    entLoss = entropy_loss(prob, logProb)
    return polLoss, entLoss


def batch_multi_agent(tensor, n_agents, use_strategies=False):
    """return tensor of shape (batch*seq_len*horizon, n_agents, features) or (n_strategies, batch*seq_len*horizon, n_agents, features)"""
    if use_strategies:
        return tensor.reshape(tensor.shape[0], -1, n_agents, tensor.shape[-1]) if tensor is not None else None
    else:
        return tensor.view(-1, n_agents, tensor.shape[-1]) if tensor is not None else None

def batch_multi_agent_horizon(tensor):
    """return tensor of shape (n_strategies, horizon, batch*seq_len, n_agent*features) used for global_trajectory_synthesizer"""

    return tensor.view(tensor.shape[0], tensor.shape[1], -1,  tensor.shape[-2]*tensor.shape[-1]) if tensor is not None else None    

def batch_strat_horizon(tensor):
    """return tensor of shape (n_strategies, horizon, batch*seq_len*n_agents, features) used for normal trajectory_synthesizer"""

    return tensor.view(tensor.shape[0], tensor.shape[1], -1, tensor.shape[-1]) if tensor is not None else None

def compute_return(reward, value, discount, bootstrap, lmbda, gamma, use_strategy_selector=False):
    if use_strategy_selector:
        next_values = torch.cat([value[:, 1:], bootstrap[:, None]], 1)
    else:
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
    target = reward + gamma * discount * next_values * (1 - lmbda)
    outputs = []
    accumulated_reward = bootstrap
    timesteps = reward.shape[1] if use_strategy_selector else reward.shape[0]
    for t in reversed(range(timesteps)):
        discount_factor = discount[:,t] if use_strategy_selector else discount[t]
        if use_strategy_selector:
            accumulated_reward = target[:, t] + gamma * discount_factor * accumulated_reward * lmbda
        else:
            accumulated_reward = target[t] + gamma * discount_factor * accumulated_reward * lmbda
        outputs.append(accumulated_reward)
    returns = torch.transpose(torch.flip(torch.stack(outputs), [0]), 0, 1) if use_strategy_selector else torch.flip(torch.stack(outputs), [0])

    return returns


def info_loss(feat, model, actions, fake):
    """loss function for maximizing InfoMax over the actions
    :param feat: features of the state
    :param model: model used to compute the action logits
    :param actions: actions taken by the agent
    :param fake: mask for the batch
    :return: loss value
    """
    q_feat = F.relu(model.q_features(feat))
    action_logits = model.q_action(q_feat)
    return (fake * action_information_loss(action_logits, actions)).mean()

def info_loss_strategy(feat, model):
    """loss function for maximizing InfoMax over the different strategies

    """
    labels = torch.arange(feat.shape[0], device=feat.device).repeat_interleave(feat.shape[1]).unsqueeze(1)
    predicted = model(feat.reshape(-1, feat.shape[-1]))
    return action_information_loss(predicted.base_dist.logits, labels).mean()


def action_information_loss(logits, target):
    criterion = nn.CrossEntropyLoss(reduction='none')
    return criterion(logits.view(-1, logits.shape[-1]), target.argmax(-1).view(-1))


def log_prob_loss(model, x, target):
    pred = model(x)
    return -torch.mean(pred.log_prob(target))


def kl_div_categorical(p, q):
    eps = 1e-7
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)


def reshape_dist(dist, config):
    return dist.get_dist(dist.deter.shape[:-1], config.N_CATEGORICALS, config.N_CLASSES)


def state_divergence_loss(prior, posterior, config, reduce=True, balance=0.2):
    prior_dist = reshape_dist(prior, config)
    post_dist = reshape_dist(posterior, config)
    post = kl_div_categorical(post_dist, prior_dist.detach())
    pri = kl_div_categorical(post_dist.detach(), prior_dist)
    kl_div = balance * post.mean(-1) + (1 - balance) * pri.mean(-1)
    if reduce:
        return torch.mean(kl_div)
    else:
        return kl_div



def info_nce_loss(traj_embed, temperature=0.05, multiple_positives=True, num_of_batches=16, seq_len= 10):
    """
    info_nce_loss function for contrastive learning without explicit labels.
    
    :params traj_embed: Input tensor of shape (labels, batch, features).
    :params temperature: Temperature parameter for softmax. Default is 0.07.
        
    :return loss: info_nce_loss loss.
    """
    # labels[i] = 0 means the first entry in logits[i] is the positive
    strat_size = traj_embed.shape[0]
    batch_size = traj_embed.shape[1]
    # Normalize input embeddings along the feature dimension
    traj_embed = F.normalize(traj_embed, dim=-1)

    # the elements coming from the same strategy can be considered as positive sample, 
    # The label matrix is a block-circulant matrix of size (batch*strategy, batch*strategy)
    # where each block is a block of ones with size batch x batch.
    # Example of labels for a batch of 3 elements and 3 strategies:
    # [[1, 1, 1, 0, 0, 0, 0, 0, 0],
    #  [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #  [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 1, 1, 1, 0, 0, 0],
    #  [0, 0, 0, 1, 1, 1, 0, 0, 0],
    #  [0, 0, 0, 1, 1, 1, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0, 1, 1, 1],
    #  [0, 0, 0, 0, 0, 0, 1, 1, 1],
    #  [0, 0, 0, 0, 0, 0, 1, 1, 1]]
    # from there we can extract the positive samples and the negative samples one per each row
    # Let's move the positive samples in the first column and the negative samples in the remaining columns
    traj_embed = torch.reshape(traj_embed, (traj_embed.shape[0]*traj_embed.shape[1],*traj_embed.shape[2:]))
    sim_matrix = torch.matmul(traj_embed, traj_embed.transpose(0,1)) # (labels*batch, labels*batch)
    label_submatrix = generate_label_submatrix(batch_size)
    positive_samples = []
    remaining_samples = []
    for strat in range(strat_size):
        if multiple_positives:
            positive_samples.append(sim_matrix[strat*batch_size:(strat+1)*batch_size, strat*batch_size:(strat+1)*batch_size][label_submatrix.to(bool)].reshape(batch_size, -1))
        else:
            positive_samples.append(sim_matrix[strat*batch_size:(strat+1)*batch_size, strat*batch_size:(strat+1)*batch_size][label_submatrix.to(bool)])
        remaining_samples.append(torch.cat([sim_matrix[strat*batch_size:(strat+1)*batch_size, :strat*batch_size], sim_matrix[strat*batch_size:(strat+1)*batch_size, (strat+1)*batch_size:]], dim=-1))
        # remaining_samples.append(sim_matrix[strat*batch_size:(strat+1)*batch_size, (strat+1)*batch_size:])
    if multiple_positives:
        positive_samples = torch.cat(positive_samples, dim=0)
    else:
        positive_samples = torch.cat(positive_samples, dim=0).unsqueeze(-1)
    remaining_samples = torch.cat(remaining_samples, dim=0)

    num_of_batches = num_of_batches
    if multiple_positives:
        loss = 0.0
        # Samples batches of positive and negative samples
        for i in range (num_of_batches):
            # Randomly sample a batch of positive samples
            indp1 = torch.randint(0, positive_samples.shape[0], (batch_size,))
            indp2 = torch.randint(0, positive_samples.shape[1], (batch_size,))
            new_positive_samples = positive_samples[indp1, indp2].unsqueeze(1)
            # Randomly sample a batch of negative samples
            indn1 = torch.randint(0, remaining_samples.shape[0], (batch_size,))
            indn2 = torch.randint(0, remaining_samples.shape[1], (batch_size,batch_size))
            new_remaining_samples = remaining_samples[indn1[:, None], indn2]
            # Concatenate the two batches
            logits = torch.cat([new_positive_samples, new_remaining_samples], dim=1)
            logits = logits / temperature
            labels = torch.zeros(batch_size, dtype=torch.long).to(traj_embed.device)
            loss += F.cross_entropy(logits, labels)
    # Concatenate positive and negative similarities
    else:
        logits = torch.cat([positive_samples, remaining_samples], dim=1)
        # Compute labels for NCE loss (positive sample has index 0)
        # zero means that the positive sample is in the first column not that the label is 0 (different from standard one-hot encoding labels)
        labels = torch.zeros(batch_size*strat_size, dtype=torch.long).to(traj_embed.device)
        
        logits = logits / temperature
        
        # Compute info_nce_loss using cross-entropy
        loss = F.cross_entropy(logits, labels)
    
    return loss


def generate_label_submatrix(N):
    """Given a number N, it generates a square matrix with 1s in the first element 
    of the last row and 1s after the main diagonal in each row except the last one.
    example:
    N = 3
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0]]
    """
    matrix = [[0] * N for _ in range(N)]

    for i in range(N - 1):
        matrix[i][i+1] = 1  # Set 1 after the main diagonal in each row except the last one
    
    # Set 1 in the first element of the last row
    matrix[N-1][0] = 1

    return torch.tensor(matrix)

# some implementation of the InfoNCE loss also includes double counting (symmetric elements (i,j) and (j,i))
# this is redundant but as pro there are more positives and as cons more computational load
# def generate_label_submatrix(N):
#     """
#     Generates a square matrix of shape (N, N) where:
#     - Diagonal = 0 (exclude self-similarity)
#     - All off-diagonal entries = 1 (all others in the group are positives)
#     """
#     matrix = torch.ones((N, N), dtype=torch.long)
#     matrix.fill_diagonal_(0)
#     return matrix

if __name__ == "__main__":
    num_strategies = 4
    batch_size = 630
    # create a tensor of size num_strategies x batch_size x 2 with increasing integers as a float32 tensor
    input_tensor = torch.arange(num_strategies * batch_size * 2, dtype=torch.float32).reshape(num_strategies, batch_size, 2)
    loss_out = info_nce_loss(input_tensor)
    print(loss_out)