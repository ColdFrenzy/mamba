import ray
import wandb
from pathlib import Path
import torch

from agent.workers.DreamerWorker import DreamerWorker


class DreamerServer:
    """The server is the entity that manages the workers, it sends the model to the workers and collects the results.
    """
    def __init__(self, n_workers, env_config, controller_config, model):
        # if local_mode=True it runs the workers in series instead of parallel
        ray.init(local_mode=True)

        self.workers = [DreamerWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
        self.tasks = [worker.run.remote(model) for worker in self.workers]
        self.eval_tasks = []

    def append(self, idx, update):
        """
        :param idx: worker index, 
        :param update: model weights used to update the workers"""
        self.tasks.append(self.workers[idx].run.remote(update))

    def append_eval(self, idx, update, n_episodes=100):
        """evaluate the model over n_episodes
        :param idx: worker index,
        :param update: model weights used to update the workers,
        """
        self.eval_tasks.append(self.workers[idx].eval.remote(update, n_episodes))

    def run(self):
        done_id, tasks = ray.wait(self.tasks)
        self.tasks = tasks
        recvs = ray.get(done_id)[0]
        return recvs

    def eval(self):
        done_id, tasks = ray.wait(self.eval_tasks)
        self.eval_tasks = tasks
        recv = ray.get(done_id)[0]
        return recv
    
class DreamerServerEval:
    """The server is the entity that manages the workers, it sends the model to the workers and collects the results.
    """
    def __init__(self, n_workers, env_config, controller_config, learner_config, model, n_episodes):
        # if local_mode=True it runs the workers in series instead of parallel
        ray.init(local_mode=True)

        self.workers = [DreamerWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
        self.eval_tasks = [worker.eval.remote(model, n_episodes, learner_config, return_all=True, return_strategy_plot=True) for worker in self.workers]

    def eval(self):
        done_id, tasks = ray.wait(self.eval_tasks)
        self.eval_tasks = tasks
        recv = ray.get(done_id)[0]
        return recv

class DreamerRunner:
    """
    Runner is the main entity that runs the training loop, it contains the learner and the server (which contains the workers).
    """
    def __init__(self, env_config, learner_config, controller_config, n_workers):
        self.n_workers = n_workers
        self.learner = learner_config.create_learner()
        self.server = DreamerServer(n_workers, env_config, controller_config, self.learner.params())
        self.env_name = env_config.to_dict()["env_configs0_env_name"]

    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10):
        cur_steps, cur_episode = 0, 0
        self.current_checkpoint = self.learner.test_every
        Path(self.learner.config.WEIGHTS_FOLDER).mkdir(parents=True, exist_ok=True)
        run_name = create_name(self.learner.config, self.env_name, max_steps)
        if self.learner.use_wandb:
            wandb.run.name = run_name
            wandb.define_metric("steps")
            wandb.define_metric("reward", step_metric="steps")
            wandb.define_metric("eval/eval_steps")
            wandb.define_metric("eval/win_rate", step_metric="eval/eval_steps")
            wandb.define_metric("eval/mean_steps", step_metric="eval/eval_steps")
            wandb.run.tags = return_tags(self.learner.config, self.env_name, max_steps)
        while True:
            rollout, info = self.server.run()
            self.learner.step(rollout)
            cur_steps += info["steps_done"]
            strat = {k: v for k, v in info.items() if 'strategy_' in k}
            cur_episode += 1
            if self.learner.use_wandb:
                wandb.log({'reward': info["reward"], 'steps': cur_steps, **strat})

            print(cur_episode, self.learner.total_samples, info["reward"])
            
            if cur_steps >= self.current_checkpoint and cur_steps < max_steps:
                current_name = create_name(self.learner.config, self.env_name, self.current_checkpoint)
                save_file_name =  current_name + ".pt"
                save_path = file_path = Path("wandb") / save_file_name
                # at the end of the training evaluate over 100 episodes
                self.server.append_eval(info['idx'], self.learner.params(), 100)
                info = self.server.eval()
                if self.learner.use_wandb:
                    wandb.log({'eval/win_rate': info['win_rate'], 'eval/mean_steps': info['mean_steps'], 'eval/eval_steps': self.current_checkpoint})
                # and save the model
                torch.save(self.learner.params(), save_path)
                self.current_checkpoint += self.learner.test_every


            if cur_episode >= max_episodes or cur_steps >= max_steps:
                current_name = create_name(self.learner.config, self.env_name, self.current_checkpoint)
                save_file_name =  current_name + ".pt"
                save_path = file_path = Path("wandb") / save_file_name
                # at the end of the training evaluate over 100 episodes
                self.server.append_eval(info['idx'], self.learner.params(), 100)
                info = self.server.eval()
                if self.learner.use_wandb:
                    wandb.log({'eval/win_rate': info['win_rate'], 'eval/mean_steps': info['mean_steps'], 'eval/eval_steps': self.current_checkpoint})
                # and save the model
                torch.save(self.learner.params(), save_path)
                break
            self.server.append(info['idx'], self.learner.params())



class DreamerEvaluator:
    def __init__(self, env_config, learner_config, controller_config, n_workers, weights_path, n_episodes):
        self.n_workers = n_workers
        self.weights_path = weights_path
        self.learner_config = learner_config
        self.weights = torch.load(weights_path)
        self.server = DreamerServerEval(n_workers, env_config, controller_config, learner_config, self.weights, n_episodes)
        self.env_name = env_config.to_dict()["env_configs0_env_name"]

    def run(self):
        """
        Load pre-trained model and run test
        """
        run_name = create_name(self.learner_config, self.env_name, eval=True)
        if self.learner_config.USE_WANDB:
            global wandb
            import wandb
            wandb.init(project= "mamba_eval", dir=self.learner_config.LOG_FOLDER, config=self.learner_config.__dict__)

        if self.learner_config.USE_WANDB:
            wandb.run.name = run_name
            # wandb.define_metric("eval/eval_steps")
            # wandb.define_metric("eval/win_rate", step_metric="eval/eval_steps")
            # wandb.define_metric("eval/mean_steps", step_metric="eval/eval_steps")
            wandb.run.tags = return_tags(self.learner_config, self.env_name, eval=True)
        while True:
            info = self.server.eval()
            strat = {k: v for k, v in info.items() if 'strategy_' in k}

            print("Evaluation finished")
            print("win_rate:", info["win_rate"], "mean_steps:", info["mean_steps"])
            # at the end of the training evaluate over n_episodes episodes
            if self.learner_config.USE_WANDB:
                for step in range(0, len(info['win_rate'])):
                    if self.learner_config.USE_STRATEGY_SELECTOR:
                        strat_at_step = {k: v[step] for k, v in strat.items()}
                        wandb.log({'eval/win': info['win_rate'][step], 'eval/ep_steps': info['mean_steps'][step], **strat_at_step})
                    else:
                        wandb.log({'eval/win': info['win_rate'][step], 'eval/mean_steps': info['mean_steps'][step]})

            # and save the model
            break



def create_name(config, env_name, max_steps=None, eval = False):
    """Return the name of the saving file based on the configuration.
    The name is composed as: "architecture_" + "env_name_" + "
    
    """

    if config.USE_COMMUNICATION:
        if eval:
            file_name = "Eval_MAMBA_"
        else:
            file_name = "MAMBA_"
    else:
        if eval:
            file_name = "Eval_MultiDreamer_"
        else:
            file_name = "MultiDreamer_"
    file_name = file_name + env_name
    if config.USE_SHARED_REWARD:
        file_name = file_name + "_SR"
    if config.USE_STRATEGY_ADVANTAGE:
        file_name = file_name + "_SA"
    if config.USE_AUGMENTED_CRITIC:
        file_name = file_name + "_AC"
    if config.USE_TRAJECTORY_SYNTHESIZER:
        file_name = file_name + "_TS"
    if config.USE_LAST_STATE_VALUE:
        file_name = file_name + "_LSV"
    if max_steps is not None:
        file_name = file_name + "_" + abbreviate_number(max_steps)

    return file_name



def return_tags(config, env_name, max_steps=None, eval=False):
    tags = []
    if config.USE_COMMUNICATION:
        tags.append("MAMBA")
    else:
        tags.append("MultiDreamer")
    tags.append(env_name)
    if config.USE_SHARED_REWARD:
        tags.append("SR")
    if config.USE_STRATEGY_ADVANTAGE:
        tags.append("SA")
    if config.USE_AUGMENTED_CRITIC:
        tags.append("AC")
    if config.USE_TRAJECTORY_SYNTHESIZER:
        tags.append("TS")
    if config.USE_LAST_STATE_VALUE:
        tags.append("LSV")
    if max_steps is not None:
        tags.append(abbreviate_number(max_steps))

    return tags



def abbreviate_number(num):
    if num >= 1_000_000_000:
        return f'{num / 1_000_000_000:.1f}B'
    elif num >= 1_000_000:
        return f'{num / 1_000_000:.1f}M'
    elif num >= 1_000:
        return f'{num / 1_000:.1f}K'
    else:
        return str(num)