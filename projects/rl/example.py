import os
import sys
from typing import Any, Dict, Tuple, Union

import mlflow
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger

dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')
load_dotenv(dotenv_path)

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("RL Example")


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

# Check and end any active runs
if mlflow.active_run() is not None:
    mlflow.end_run()

with mlflow.start_run():
    print("Training model...")
    model = PPO("MlpPolicy", vec_env, verbose=2)
    model.set_logger(
        Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )
    )
    model.learn(total_timesteps=25000, log_interval=1)
    print("Training completed")
