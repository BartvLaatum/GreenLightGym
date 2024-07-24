from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO
from greenlight_gym.experiments.utils import load_env_params, make_vec_env, load_model_params, wandb_init, create_callbacks
from greenlight_gym.common.results import Results
# from greenlight_gym.envs.greenlight import GreenLightHeatCO2
import wandb

if __name__ == '__main__':

    SEED = 666
    project= 'continue_training'
    runname = 'gallant-firefly-1'
    path = f'train_data/{project}/models/'
    config_path = 'configs/'
    config_name = 'benchmark_mutliplicative_pen'
    group = 'test'
    job_type = f'seed-{SEED}'

    env_id = 'GreenLightHeatCO2'
    vec_norm_kwargs = {"norm_obs": True, "norm_reward": True, "clip_obs": 50_000}

    env_base_params, env_specific_params, options, result_columns = load_env_params(env_id, config_path+'envs/', config_name)
    model_params = load_model_params('ppo-98', config_path+'algorithms/', 'multiplicative_pen')
    results = Results(result_columns)

    n_eval_episodes = 10
    n_envs = 12
    total_timesteps = 100_000
    n_evals = 2
    eval_freq = total_timesteps//n_evals//n_envs
    run_id= 'ql70rqoz'

    run, config = wandb_init(model_params, env_base_params, env_specific_params, total_timesteps, SEED, project=project, group=group, runname='continued-' + runname, job_type=job_type, save_code=True)

    env = make_vec_env(env_id,
                    env_base_params,
                    env_specific_params,
                    options,
                    seed=SEED,
                    n_envs=12,
                    monitor_filename=None,
                    vec_norm_kwargs=vec_norm_kwargs,
                    eval_env=False)

    eval_env = make_vec_env(env_id,
                    env_base_params,
                    env_specific_params,
                    options,
                    seed=SEED,
                    n_envs=min(n_eval_episodes, 10),
                    monitor_filename=None,
                    vec_norm_kwargs=vec_norm_kwargs,
                    eval_env=True)

    model_log_dir = f"train_data/{project}/models/{runname}/"
    env_log_dir = f"train_data/{project}/envs/{runname}/"

    env = VecNormalize.load(f'train_data/{project}/envs/{runname}/vecnormalize.pkl', env)
    eval_env = VecNormalize.load(f'train_data/{project}/envs/{runname}/vecnormalize.pkl', eval_env)

    callbacks = create_callbacks(
        n_eval_episodes,
        eval_freq,
        env_log_dir,
        save_name=None,
        model_log_dir=model_log_dir,
        eval_env=eval_env,
        run=run,
        results=results,
        save_env=None,
        verbose=1
        )

    tensorboard_log = f"train_data/{project}/logs/{runname}"
    model = PPO.load(path + f'{runname}/best_model.zip', env=env, tensorboard_log=tensorboard_log, verbose=1)

    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)
    # print(model.num_timesteps)
