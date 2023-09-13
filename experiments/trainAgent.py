from stable_baselines3 import PPO
from RLGreenLight.environments.GreenLight import GreenLight
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

import yaml
from wandb.integration.sb3 import WandbCallback
import wandb
from RLGreenLight.callbacks.customCallback import TensorboardCallback
from stable_baselines3.common.utils import set_random_seed
from torch.optim import Adam
from torch.nn.modules.activation import ReLU, SiLU
from multiprocessing import cpu_count

ACTIVATION_FN = {"ReLU": ReLU, "SiLU": SiLU}
OPTIMIZER = {"ADAM": Adam}

if __name__ == "__main__":
    # print(make_vec_env(env_id="Pendulum-v1", n_envs=2, seed=666, start_index=0))    # hyperparameters
    with open("hyperparameters/ppo/balance-rew-no-constraints.yml", "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    envParams = params["GreenLight"]
    modelParams = params["PPO"]
    options = params["options"]

    # set random seed

    # check if policy kwargs are given
    print("policy_kwargs" in modelParams.keys())
    if "policy_kwargs" in modelParams.keys():
        modelParams["policy_kwargs"]["activation_fn"] = ACTIVATION_FN[modelParams["policy_kwargs"]["activation_fn"]]
        modelParams["policy_kwargs"]["optimizer_class"] = OPTIMIZER[modelParams["policy_kwargs"]["optimizer_class"]]

    numCpus = cpu_count() - 2
    SEED = 666
    config= {
        "policy": MlpPolicy,
        "total_timesteps": 300_000,
        "env": GreenLight(**envParams),
        "eval_env": GreenLight(**envParams, options=options, training=False),
        "seed": SEED,
        "note": "testing co2 control, larger nets, smaller pred horizon",
        "modelParams": {**modelParams}
    }

    run = wandb.init(
        project="RLGreenLight",
        config=config,
        group="PPO-reward-balance",
        sync_tensorboard=True,
        job_type="train",
        save_code=True
    )

    env = DummyVecEnv([lambda: GreenLight(**envParams) for _ in range(numCpus)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=1000, clip_obs=100)
    env = VecMonitor(env, filename=None)

    eval_env = DummyVecEnv([lambda: config["eval_env"]])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_reward=1000, clip_obs=100)
    eval_env = VecMonitor(eval_env, filename=None)

    evalCallback = TensorboardCallback(eval_env=eval_env, n_eval_episodes=1, eval_freq=config["total_timesteps"]//10//numCpus, log_path=None, best_model_save_path=f"./trainData/models/{run.name}", deterministic=True, verbose=1)
    wandbcallback = WandbCallback(verbose=1)

    tensorboard_log = f"trainData/logs/{run.name}"

    model = PPO(env=env, seed=SEED, verbose=0, **config["modelParams"], tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=config["total_timesteps"], 
                callback=[wandbcallback, evalCallback])
    env.save("trainData/environments/testenv.p")
    model.save("trainData/models/finalmodel.p")
    run.finish()
