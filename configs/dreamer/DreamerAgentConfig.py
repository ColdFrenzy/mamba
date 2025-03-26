from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.Config import Config

RSSM_STATE_MODE = 'discrete'


class DreamerConfig(Config):
    def __init__(self):
        super().__init__()
        self.USE_TEST_CONFIG = True # use a small config for testing
        self.USE_TRAJECTORY_SYNTHESIZER = True
        self.USE_GLOBAL_TRAJECTORY_SYNTHESIZER = True # use a global trajectory synthesizer
        self.USE_STRATEGY_SELECTOR = True
        self.USE_COMMUNICATION = True # learn communication and use it also in the imagination
        self.USE_STRATEGY_ADVANTAGE = False
        self.USE_SHARED_REWARD = True
        self.USE_AUGMENTED_CRITIC = True
        self.USE_WANDB = True
        self.USE_LAST_STATE_VALUE = False # whether to use a value function for the last state in the trajectory or the average value
        self.TEST_EVERY =  5 if self.USE_TEST_CONFIG else 10000 # every ~10000 steps test the model for evaluating performances
        self.STRATEGY_DURATION = 5 if self.USE_TEST_CONFIG else 15
        self.N_STRATEGIES = 2 if self.USE_TEST_CONFIG else 4
        self.HORIZON = 5 if self.USE_TEST_CONFIG else 15
        self.HIDDEN = 64 if self.USE_TEST_CONFIG else 256
        self.MODEL_HIDDEN = 64 if self.USE_TEST_CONFIG else 256
        self.EMBED = 64 if self.USE_TEST_CONFIG else 256
        self.N_CATEGORICALS = 10 if self.USE_TEST_CONFIG else 32
        self.N_CLASSES = 10 if self.USE_TEST_CONFIG else 32
        self.STOCHASTIC = self.N_CATEGORICALS * self.N_CLASSES
        self.DETERMINISTIC = 64 if self.USE_TEST_CONFIG else 256
        self.FEAT = self.STOCHASTIC + self.DETERMINISTIC
        if self.USE_STRATEGY_SELECTOR:
            self.ACTOR_FEAT = self.FEAT + (self.N_STRATEGIES-1) if self.N_STRATEGIES > 1 else self.FEAT + 1
        else:
            self.ACTOR_FEAT = self.FEAT
        self.GLOBAL_FEAT = self.FEAT + self.EMBED
        self.TRAJECTORY_SYNTHESIZER_LAYERS = 4
        self.TRAJECTORY_SYNTHESIZER_HIDDEN = 64 if self.USE_TEST_CONFIG else 256
        self.TRAJECTORY_SYNTHESIZER_HEADS = 1 if self.USE_TEST_CONFIG else 8 # if using transformer
        self.VALUE_LAYERS = 2
        self.VALUE_HIDDEN = 64 if self.USE_TEST_CONFIG else 256
        self.PCONT_LAYERS = 2
        self.PCONT_HIDDEN = 64 if self.USE_TEST_CONFIG else 256
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 2
        self.ACTION_HIDDEN = 64 if self.USE_TEST_CONFIG else 256
        self.REWARD_LAYERS = 2
        self.REWARD_HIDDEN = 64 if self.USE_TEST_CONFIG else 256
        self.GAMMA = 0.99
        self.DISCOUNT = 0.99
        self.DISCOUNT_LAMBDA = 0.95
        self.IN_DIM = 30


@dataclass
class RSSMStateBase:
    stoch: torch.Tensor
    deter: torch.Tensor

    def map(self, func):
        return RSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        pass


@dataclass
class RSSMStateDiscrete(RSSMStateBase):
    logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)


@dataclass
class RSSMStateCont(RSSMStateBase):
    mean: torch.Tensor
    std: torch.Tensor

    def get_dist(self, *input):
        return td.independent.Independent(td.Normal(self.mean, self.std), 1)


RSSMState = {'discrete': RSSMStateDiscrete,
             'cont': RSSMStateCont}[RSSM_STATE_MODE]
