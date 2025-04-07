import json

from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from agent.utils.paths import WEIGHTS_DIR


def save_full_config(configs, save_file):
    """Save the full configuration (all classes) to a JSON file."""
    full_config_dict = {
        'DreamerControllerConfig': vars(configs['controller_config']),
        'DreamerLearnerConfig': vars(configs['learner_config']),
    }
    
    with open(save_file, 'w') as config_file:
        json.dump(full_config_dict, config_file, indent=4)
    
    print(f"Full configuration saved to {save_file}")

def load_full_config(config_classes, config_file_path, excluded_keys=None):
    """Load the full configuration and reinitialize classes."""
    with open(config_file_path, 'r') as config_file:
        full_config_dict = json.load(config_file)

    # Reinitialize the classes using the loaded data
    controller_config = config_classes['controller']
    learner_config = config_classes['learner']

    # Set attributes from the loaded config
    for key, value in full_config_dict['DreamerControllerConfig'].items():
        setattr(controller_config, key, value)
    
    for key, value in full_config_dict['DreamerLearnerConfig'].items():
        if excluded_keys and key in excluded_keys:
            continue
        setattr(learner_config, key, value)
    

    return controller_config, learner_config

ONLY_WM_EXCLUDED_KEYS = ["ACTOR_LR", "VALUE_LR", "USE_TEST_CONFIG", "USE_TRAJECTORY_SYNTHESIZER", "USE_GLOBAL_TRAJECTORY_SYNTHESIZER",
                         "USE_STRATEGY_SELECTOR","USE_STRATEGY_ADVANTAGE", "USE_AUGMENTED_CRITIC", "USE_WANDB", "USE_LAST_STATE_VALUE",
                         "TEST_EVERY", "STRATEGY_DURATION", "N_STRATEGIES", "HORIZON",
                         "ACTOR_FEAT", "TRAJECTORY_SYNTHESIZER_LAYERS", "TRAJECTORY_SYNTHESIZER_HIDDEN", "TRAJECTORY_SYNTHESIZER_HEADS",
                         "VALUE_LAYERS", "VALUE_HIDDEN","ACTION_SIZE", "ACTION_LAYERS", "ACTION_HIDDEN", "GAMMA", "DISCOUNT", "DISCOUNT_LAMBDA",
                        "TRAJECTORY_SYNTHESIZER_LR", "EPOCHS", "PPO_EPOCHS","BATCH_SIZE", "SEQ_LENGTH", 
                        "TARGET_UPDATE","GRAD_CLIP", "ENTROPY", "ENTROPY_ANNEALING", "GRAD_CLIP_POLICY", "TRAJECTORY_SYNTHESIZER_SCALE"]
                        

if __name__ == "__main__":
    controller_config = DreamerControllerConfig()
    learner_config = DreamerLearnerConfig()

    configs = {
        'controller_config': controller_config,
        'learner_config': learner_config,
    }

    save_full_config(configs, 'experiment_directory')
    # Example usage
    loaded_controller_config, loaded_learner_config = load_full_config({
        'controller': DreamerControllerConfig(),
        'learner': DreamerLearnerConfig(),
    }, WEIGHTS_DIR / 'full_config.json')

    # instantiate a new controller and learner config with the loaded data
    controller_config = loaded_controller_config
    learner_config = loaded_learner_config

    print(controller_config)
    print(learner_config)

