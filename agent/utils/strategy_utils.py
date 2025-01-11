import torch
import matplotlib.pyplot as plt
from agent.optim.loss import actor_rollout
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

def encode_strategy(strategy, n_strategies):
    """return a one hot encoding representation from a strategy index
    :param strategy: torch.Tensor(n_agents)
    :return: torch.Tensor(n_agents, n_strategies-1)
    """

    strategy_encoded = torch.zeros(strategy.size(0), n_strategies-1) if n_strategies > 1 else torch.zeros(strategy.size(0), 1)
    for i in range(strategy.size(0)):
        strategy_encoded[i, strategy[i]-1] = 1
    return strategy_encoded

def generate_trajectory_scatterplot(embed_traj):
    """
    Given multiple trajectories generated from different strategies and embedded in a lower dimensional space, this function returns 
    an image that represents their distribution in a 2D space (by picking the first two features).
    alternatively we could use t-distributed stochastic neighbor embedding (t-SNE)
    :params embed_traj: [num_strategies, embed_dim] 
    :return traj_img: pyplot image
    """
    reduced_embed_traj = embed_traj[:,:,:2].detach().clone().cpu()
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(reduced_embed_traj.shape[0]):
        ax.scatter(reduced_embed_traj[i, :, 0], reduced_embed_traj[i, :, 1], label=f'Strategy {i}')
    plt.title('Trajectories scatterplot in 2D Space')
    plt.legend()

    # gcf gets the current figure
    traj_img = plt.gcf()
    
    plt.close(fig)

    return traj_img

def generate_trajectories(rollouts, model, actor, critic, config, trajectory_synthesizer, use_wandb=False):
    """Generate trajectories from the current policy and visualize them in a 2D space.
    :param rollouts: dictionary containing the rollouts
    :param model: the model used for the imagination
    :param actor: the actor used for the imagination
    :param critic: the critic used for the imagination
    :param config: the configuration object
    :param trajectory_synthesizer: the trajectory synthesizer model
    """

    samples = rollouts
    actions, av_actions, old_policy, imag_feat, returns = actor_rollout(samples['observation'],
                                                                samples['action'],
                                                                samples['last'],
                                                                model,
                                                                actor,
                                                                critic,
                                                                config,
                                                                samples['neighbors_mask'],
                                                                detach_results=False)
    # trajectories [n_strategies, horizon, n_agents*batch_size*seq_len, n_features]
    trajectories = torch.cat([imag_feat, actions], dim=-1)
    traj_embed = []
    for traj in range(len(trajectories)):
        traj_embed.append(trajectory_synthesizer(trajectories[traj]))

    traj_embed = torch.stack(traj_embed, dim=0)
    # traj_embed_fig = generate_trajectory_scatterplot(traj_embed)
    reshaped_traj_embed = traj_embed.reshape(-1, traj_embed.shape[-1])
    labels = torch.zeros(reshaped_traj_embed.shape[0])
    for i in range(traj_embed.shape[0]):
        labels[i*reshaped_traj_embed.shape[0]//traj_embed.shape[0]:(i+1)*reshaped_traj_embed.shape[0]//traj_embed.shape[0]] = i
    plt = use_pca(reshaped_traj_embed, labels)
    # plt.show()
    pl2 = use_tsne(reshaped_traj_embed, labels)
    # pl2.show()
    pl3 = use_umap(reshaped_traj_embed, labels)
    # pl3.show()
    # if self.use_wandb and np.random.randint(100) == 9:
    #     wandb.log({'Plots/Trajectory_Embeddings': wandb.Image(traj_embed_fig)})



def use_pca(tensor, labels):
    # Assuming `tensor` is your N-dimensional tensor and `labels` are the corresponding class labels
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tensor)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Classes')
    plt.title('PCA of Contrastive Learning Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()



def use_tsne(tensor, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(tensor)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Classes')
    plt.title('t-SNE of Contrastive Learning Results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


def use_umap(tensor, labels):
    umap_result = umap.UMAP(n_components=2).fit_transform(tensor)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Classes')
    plt.title('UMAP of Contrastive Learning Results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()




if __name__ == "__main__":
    import numpy as np
    N = 100
    D = 50
    tensor = np.random.randn(N, D)
    labels = np.random.randint(0, 5, size=N)


    use_pca(tensor, labels)
    use_tsne(tensor, labels)
    use_umap(tensor, labels)
    print("you are in the main function")
