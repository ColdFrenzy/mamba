import argparse
import os
import ray

from agent.runners.DreamerRunner import DreamerRunner
from agent.utils.paths import  STARCRAFT_DIR
from agent.utils.save_utils import load_full_config, ONLY_WM_EXCLUDED_KEYS
from configs import Experiment
from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig, GridWorldConfig

from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from environments import Env


USE_RAY = False
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["SC2PATH"] = os.path.join(current_dir,"env", "starcraft")
# ray.init(
#     runtime_env={
#         "env_vars": {"RAY_DEBUG": "1", "SC2PATH": os.environ["SC2PATH"]},
#     }
# )
# 5GB memory for each worker
if USE_RAY:
    ray.init()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="starcraft", help='starcraft or GridWorld env')
    parser.add_argument('--env_name', type=str, default="3m", help='Specific setting')
    parser.add_argument('--n_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--continue_training', type=bool, default=False, help='Continue training')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load the full model (agent + world model)')
    parser.add_argument('--wm_path', type=str, default=None, help='Path to load only the world model')
    parser.add_argument('--ac_path', type=str, default=None, help='Path to load only the actor-critic')
    return parser.parse_args()


def train_dreamer(exp, n_workers, use_ray=True):
    runner = DreamerRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers, random_seed=exp.random_seed, use_ray=use_ray)
    runner.run(exp.steps, exp.episodes)


def get_env_info(configs, env):
    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = int(env.n_actions)
    env.close()


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}


def prepare_gridworld_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = GridWorldConfig(env_name)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def single_run(args, random_seed=23):

    RANDOM_SEED = random_seed
    args = args
    if args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    elif args.env == Env.GRIDWORLD:
        configs = prepare_gridworld_configs(args.env_name)
    else:
        raise Exception("Unknown environment")
    if args.wm_path is not None:
        excluded_keys = ONLY_WM_EXCLUDED_KEYS
        configs["learner_config"], configs["controller_config"] = load_full_config(configs, args.wm_path, excluded_keys)
        configs["learner_config"].MODEL_EPOCHS = 0 # no training for the world model
    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    exp = Experiment(steps= 100 if configs["learner_config"].USE_TEST_CONFIG else 100000,
                     episodes= 5 if configs["learner_config"].USE_TEST_CONFIG else 50000,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])

    train_dreamer(exp, n_workers=args.n_workers, use_ray=USE_RAY)


if __name__ == "__main__":
    args = parse_args()
    for seed in [23, 95, 247]:
        single_run(args, random_seed=seed)
