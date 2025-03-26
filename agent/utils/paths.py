import pathlib


ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
STARCRAFT_DIR = ROOT_DIR / "env" / "starcraft"
LOG_DIR = ROOT_DIR / "wandb" # create if not exists
LOG_DIR.mkdir(exist_ok=True, parents=True)
