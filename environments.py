from enum import Enum


class Env(str, Enum):
    STARCRAFT = "starcraft"
    GRIDWORLD = "gridworld"




RANDOM_SEED = 23
ENV = Env.STARCRAFT
ENV_NAME = "3m"
