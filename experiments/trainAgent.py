from RLGreenLight.experiments.utils import loadParameters, wandb_init, make_vec_env, create_callbacks
from stable_baselines3 import PPO
from multiprocessing import cpu_count


if __name__ == "__main__":
    hpPath = "hyperparameters/ppo/"
    filename = "balance-rew-no-constraints.yml"
    total_timesteps = 10_000
    numCpus = cpu_count() - 2
    SEED = 666
    project = "TestVecLoadSave"
    group = "short-test"
    envParams, modelParams, options = loadParameters(hpPath, filename)

    # define
    run, config = wandb_init(modelParams, envParams, options, total_timesteps, SEED, project=project, group=group, job_type=None, save_code=True)
    vec_norm_kwargs = {"norm_obs": True, "norm_reward": False, "clip_obs": 1000}
    monitor_filename = None

    env = make_vec_env(config["env"], numCpus=numCpus, monitor_filename=monitor_filename, vec_norm_kwargs=vec_norm_kwargs)
    eval_env = make_vec_env(config["eval_env"], numCpus=1, monitor_filename=monitor_filename, vec_norm_kwargs=vec_norm_kwargs, eval_env=True)


    env_log_dir = f"trainData/{project}/envs/{run.name}/"
    model_log_dir = f"trainData/{project}/models/{run.name}/"
    n_evals= 2
    eval_freq = total_timesteps//n_evals//numCpus
    save_name = "vec_norm"

    callbacks = create_callbacks(eval_freq, env_log_dir, save_name, model_log_dir, eval_env, verbose=1)

    tensorboard_log = f"trainData/{project}/logs/{run.name}"

    model = PPO(env=env, seed=SEED, verbose=0, **config["modelParams"], tensorboard_log=tensorboard_log)
    print("LEARNING")
    model.learn(total_timesteps=config["total_timesteps"], 
                callback=callbacks)
    run.finish()
