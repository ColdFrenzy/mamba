import argparse
import os

from agent.runners.DreamerRunner import DreamerEvaluator
from agent.utils.paths import  STARCRAFT_DIR, WEIGHTS_DIR
from configs import Experiment, EvalExperiment, SimpleObservationConfig, NearRewardConfig, DeadlockPunishmentConfig, RewardsComposerConfig
from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig
from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from environments import Env

USE_RAY = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="flatland", help='Flatland or SMAC env')
    parser.add_argument('--env_name', type=str, default="5_agents", help='Specific setting')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to weights')
    return parser.parse_args()


def eval_dreamer(exp, n_workers, use_ray=True):
    runner = DreamerEvaluator(exp.env_config, exp.learner_config, exp.controller_config, n_workers, exp.weights_path, exp.episodes, use_ray=use_ray)
    runner.run()


def get_env_info(configs, env):
    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
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



if __name__ == "__main__":
    RANDOM_SEED = 23
    args = parse_args()
    if args.env == Env.STARCRAFT:
        os.environ["SC2PATH"] = STARCRAFT_DIR
        configs = prepare_starcraft_configs(args.env_name)
    else:
        raise Exception("Unknown environment")
    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    exp = EvalExperiment(episodes=5,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"],
                     weights_path=WEIGHTS_DIR
                     )

    # load weights and model config
    
    eval_dreamer(exp, n_workers=args.n_workers, use_ray=USE_RAY)