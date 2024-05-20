from agent.learners.DreamerLearner import DreamerLearner
from configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerLearnerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_LR = 2e-4
        self.ACTOR_LR = 5e-4
        self.VALUE_LR = 5e-4
        self.TRAJECTORY_SYNTHESIZER_LR = 4e-4
        self.CAPACITY = 100000 if self.USE_TEST_CONFIG else 250000
        self.MIN_BUFFER_SIZE = 100 if self.USE_TEST_CONFIG else 500
        self.MODEL_EPOCHS = 2 if self.USE_TEST_CONFIG else 60
        self.EPOCHS = 1 if self.USE_TEST_CONFIG else 4 
        self.PPO_EPOCHS = 2 if self.USE_TEST_CONFIG else 5
        self.MODEL_BATCH_SIZE = 15 if self.USE_TEST_CONFIG else 40
        self.BATCH_SIZE = 15 if self.USE_TEST_CONFIG else 40
        self.SEQ_LENGTH = 7 if self.USE_TEST_CONFIG else 20
        self.N_SAMPLES = 1
        self.TARGET_UPDATE = 1
        self.DEVICE = 'cpu' # cuda
        self.GRAD_CLIP = 100.0
        self.HORIZON = 5 if self.USE_TEST_CONFIG else 15
        self.ENTROPY = 0.001
        self.ENTROPY_ANNEALING = 0.99998
        self.GRAD_CLIP_POLICY = 100.0
        self.TRAJECTORY_SYNTHESIZER_SCALE = 0.1

    def create_learner(self):
        return DreamerLearner(self)
