import os
from typing import Optional, Union

import wandb
import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from RLGreenLight.experiments.evaluation import evaluate_policy
from RLGreenLight.environments.pyutils import days2date

class TensorboardCallback(EvalCallback):
    """
    Callback that logs specified data from RL model and environment to Tensorboard.
    Is a daughter class of EvalCallback from Stable Baselines3.
    Saves the best model according to the mean reward on a evaluation environment.
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        path_vec_env: Optional[str] = None,         # from where to load in VecNormalize
        name_vec_env: Optional[str] = None,         # name of the VecNormalize file
        callback_on_new_best = None,                # callback to call when a new best model is found
        run: Optional[wandb.run] = None,
        verbose: int = 1,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            callback_on_new_best=callback_on_new_best,
            verbose=verbose,
        )
        self.path_vec_env = path_vec_env
        self.name_vec_env = name_vec_env
        self.run = run

    def _on_step(self) -> bool:

        continue_training = True

        if self.n_calls % self.eval_freq == 0:
            path = os.path.join(self.path_vec_env, f"{self.name_vec_env}_{self.num_timesteps}_steps.pkl")
            self.eval_env = VecNormalize.load(path, self.eval_env)
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, episode_actions, episode_obs, time_vec = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_actions = np.mean(episode_actions, axis=0)
            mean_obs = np.mean(episode_obs, axis=0)
            
            actions = pd.DataFrame(data=mean_actions, columns=["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uIntLamp", "uGroPipe", "uBlScr"])
            actions["Time"] = pd.to_datetime(days2date(time_vec[1:], "01-01-0001")).tz_localize('CET')

            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            if self.run:
                print(actions["Time"])
                # data = [mean_actions[:,1]]
                table = wandb.Table(dataframe=actions)#, columns=["Time", "CO2 injection"])
                # self.run.log({'controls': actions})
                # Create a line plot, specifying the x-axis as the 'Time' column
                plot = wandb.plot.line(table, x="Time", y="uCO2", title='CO2')

                # Log the custom plot
                self.run.log({"Action": plot})

                # self.run.log({"my_custom_plot": wandb.plot.line(table, "Time", "uCO2", title="CO2")})

        return continue_training

class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps
    FROM STABLE BASELINES3 RLZOO LIBRARY: 
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/f6b3ff70b13d2c2156b3e0faf9994c107c649c82/utils/callbacks.py#L55
    
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print("-----------------------")
                    print(f"Saving VecNormalize to {path}")
                    print("-----------------------")

        return True
