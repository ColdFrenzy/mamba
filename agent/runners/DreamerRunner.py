import ray
import wandb

from agent.workers.DreamerWorker import DreamerWorker


class DreamerServer:
    def __init__(self, n_workers, env_config, controller_config, model):
        ray.init(local_mode=True)

        self.workers = [DreamerWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
        self.tasks = [worker.run.remote(model) for worker in self.workers]

    def append(self, idx, update):
        """
        :param idx: worker index, 
        :param update: model weights used to update the workers"""
        self.tasks.append(self.workers[idx].run.remote(update))

    def run(self):
        done_id, tasks = ray.wait(self.tasks)
        self.tasks = tasks
        recvs = ray.get(done_id)[0]
        return recvs


class DreamerRunner:

    def __init__(self, env_config, learner_config, controller_config, n_workers):
        self.n_workers = n_workers
        self.learner = learner_config.create_learner()
        self.server = DreamerServer(n_workers, env_config, controller_config, self.learner.params())

    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10):
        cur_steps, cur_episode = 0, 0

        if self.learner.use_wandb:
            wandb.define_metric("steps")
            wandb.define_metric("reward", step_metric="steps")

        while True:
            # TODO: vedere da qui come modificare gli steps per aggiungere le strategie.
            rollout, info = self.server.run()
            self.learner.step(rollout)
            cur_steps += info["steps_done"]
            cur_episode += 1
            if self.learner.use_wandb:
                wandb.log({'reward': info["reward"], 'steps': cur_steps})

            print(cur_episode, self.learner.total_samples, info["reward"])
            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break
            self.server.append(info['idx'], self.learner.params())

