from configs.Config import Config


class Experiment(Config):
    def __init__(self, steps, episodes, random_seed, env_config, controller_config, learner_config):
        super(Experiment, self).__init__()
        self.steps = steps
        self.episodes = episodes
        self.random_seed = random_seed
        self.env_config = env_config
        self.controller_config = controller_config
        self.learner_config = learner_config


class EvalExperiment(Config):
    def __init__(self, episodes, random_seed , env_config, controller_config, learner_config, weights_path):
        super(EvalExperiment, self).__init__()
        self.weights_path = weights_path
        self.episodes = episodes
        self.random_seed = random_seed
        self.env_config = env_config
        self.controller_config = controller_config
        self.learner_config = learner_config
