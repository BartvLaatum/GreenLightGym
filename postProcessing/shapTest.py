import os
import argparse

import numpy as np
import pandas as pd
import shap
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_device

from RLGreenLight.experiments.utils import loadParameters, make_vec_env
from RLGreenLight.experiments.evaluation import evaluate_policy
import matplotlib.pyplot as plt

### Latex font in plots
plt.rcParams['font.serif'] = "cmr10"
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.size'] = 24

plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('axes', unicode_minus=False)

class PolicyClass(th.nn.Module):

    def __init__(
            self,
            policy_net,
            action_net
            ):
        super().__init__()        
        self.policy_net = policy_net
        self.action_net = action_net
    
    def forward(self, features: th.Tensor) -> th.Tensor:
        latent_pi = self.policy_net(features)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        mean_actions = self.action_net(latent_pi)
        return mean_actions



if __name__ == "__main__":
    controls = {0: "uBoil", 1: "uCo2"}
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="TestVecLoadSave")
    parser.add_argument("--runname", type=str, default="polar-dew-7")
    parser.add_argument("--env_id", type=str, default="GreenLightBase")
    parser.add_argument("--HPfolder", type=str, default="GLBase")
    parser.add_argument("--HPfilename", type=str, default="ppo.yml")
    args = parser.parse_args()

    # hyperparameters
    algorithm = "PPO"
    HPpath = f"hyperparameters/{args.HPfolder}/"
    envParams, envSpecificParams, modelParams, options, state_columns, action_columns =\
                            loadParameters(args.env_id, HPpath, args.HPfilename, algorithm)
    SEED = 666

    vec_norm_kwargs = {"norm_obs": True, "norm_reward": False, "clip_obs": 50_000, "clip_reward": 1000}

    env = make_vec_env(args.env_id, envParams, envSpecificParams, options, seed=SEED, numCpus=1,\
                             monitor_filename=None, vec_norm_kwargs=vec_norm_kwargs, eval_env=True)
    env = VecNormalize.load(f"trainData/{args.project}/envs/{args.runname}/vecnormalize.pkl", env)
    
    names = ["pious-wind-5"]
    labels = ["Scaled reward"]
    markers = ["o", "^", "v"]
    models = [PPO.load(f"trainData/{args.project}/models/{args.runname}/best_model.zip", env=env) for runname in names]
    control_indices = env.get_attr("controlIdx", [0])[0]
    
    for i, model in enumerate(models[:]):
        episode_rewards, episode_lengths, episode_actions, model_actions, episode_obs, timevec = \
                                                            evaluate_policy(
                                                                model,
                                                                env,
                                                                n_eval_episodes= 1,
                                                                deterministic = True,
                                                                render= False,
                                                                callback = None,
                                                                return_episode_rewards = True,
                                                                warn =  True,
                                                                )

        meanActions = np.mean(episode_actions, axis=0)
        model_actions = np.mean(model_actions, axis=0)
        meanObs = np.mean(episode_obs, axis=0)
        states = pd.DataFrame(data=meanObs[:-1, :envParams["modelObsVars"]], columns=state_columns)
        controlSignals = pd.DataFrame(data=meanActions, columns=action_columns)

        device = get_device("auto")
        state_log_norm = env.normalize_obs(meanObs)
        state_log_norm = th.FloatTensor(state_log_norm).to(device)
        policy = PolicyClass(model.policy.mlp_extractor.policy_net, model.policy.action_net)
        explainer = shap.DeepExplainer(policy, state_log_norm)

        shap_values = explainer.shap_values(state_log_norm) # Calculate shap values

        print(shap_values[0])
        print(len(shap_values))
        print(shap_values[0].shape)
        len_x, nu = model_actions.shape
        for j in range(nu):
            cmap = 'coolwarm'
            norm = plt.Normalize(vmin=-1, vmax=1) # define color scala between -1 and +1 (like the agents action space)
            fig = plt.figure(figsize=(15,12))
            gs = fig.add_gridspec(7, hspace=0)
            axs = gs.subplots(sharex=True, sharey=False)

            axs[0].scatter(range(0, len_x), model_actions[:,j], c=model_actions[:,j], cmap = cmap, norm=norm, label=labels[i], marker=markers[i], alpha=0.8)
            axs[0].set_ylabel(controls[control_indices[j]])

            axs[1].scatter(range(0,len_x+1), meanObs[:,0], c=shap_values[j][:,0], cmap=cmap, norm=norm, label=labels[i], marker=markers[i], alpha=0.8)
            axs[1].set_ylabel("Air temp")

            axs[2].scatter(range(0,len_x+1), meanObs[:,1], c=shap_values[j][:,1], cmap=cmap, norm=norm, label=labels[i], marker=markers[i], alpha=0.8)
            axs[2].set_ylabel("CO2 conc")

            axs[3].scatter(range(0,len_x+1), meanObs[:,2], c=shap_values[j][:,2], cmap=cmap, norm=norm, label=labels[i], marker=markers[i], alpha=0.8)
            axs[3].set_ylabel("Humidity")

            axs[4].scatter(range(0,len_x+1), meanObs[:,3], c=shap_values[j][:,4], cmap=cmap, norm=norm, label=labels[i], marker=markers[i], alpha=0.8)
            axs[4].set_ylabel("Weight")
            
            axs[5].scatter(range(0,len_x+1), meanObs[:,4], c=shap_values[j][:,4], cmap=cmap, norm=norm, label=labels[i], marker=markers[i], alpha=0.8)
            axs[5].set_ylabel("Harvest")

            axs[6].scatter(range(0,len_x+1), meanObs[:,5], c=shap_values[j][:,5], cmap=cmap, norm=norm, label=labels[i], marker=markers[i], alpha=0.8)
            axs[6].set_ylabel("PAR")
            axs[0].legend()
            plt.show()
