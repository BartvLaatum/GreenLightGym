import argparse
import pandas as pd
import matplotlib.pyplot as plt
from RLGreenLight.visualisations.createFigs import createDistFig, plotDistributions
from stable_baselines3.common.vec_env import VecNormalize
from RLGreenLight.experiments.utils import loadParameters, make_vec_env
from RLGreenLight.environments.GreenLight import GreenLight
import numpy as np

def load_env(envParams, options):
    vec_norm_kwargs = {"norm_obs": True, "norm_reward": False, "clip_obs": 1000, "clip_reward": 1000}
    env = make_vec_env(lambda: GreenLight(**envParams, options=options), numCpus=1, monitor_filename=None, vec_norm_kwargs=vec_norm_kwargs, eval_env=True)
    env = VecNormalize.load(f"trainData/{args.project}/envs/{args.runname}/vecnormalize.pkl", env)
    return env

# (obs * np.sqrt(obs_rms.var + self.epsilon)) + obs_rms.mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20111001", help="Starting date of the simulation")
    parser.add_argument("--project", type=str, default="TestVecLoadSave")
    parser.add_argument("--seasonLength", type=int, default=120, help="Length of the season")
    parser.add_argument("--controller", type=str, default="ppo", help="Controller to compare")
    parser.add_argument("--runname",  type=str, help="Runname of the controller")
    parser.add_argument("--months", nargs="*", type=int, default=[10], help="Month to plot")
    parser.add_argument("--days", nargs="*", type=int, default=None, help="Days to plot")
   
    args = parser.parse_args()
   
    hpPath = "hyperparameters/ppo/"
    filename = "balance-rew-no-constraints.yml"
    envParams, modelParams, options = loadParameters(hpPath, filename)

    env = load_env(envParams=envParams, options=options)

    # load data rule based controller
    states = pd.read_csv(f"data/{args.controller}/{args.runname}/states{args.date}-{args.seasonLength:03}.csv")
    controls = pd.read_csv(f"data/{args.controller}/{args.runname}/controls{args.date}-{args.seasonLength:03}.csv")
    states2plot = states.columns[1:]
    # env.norm_obs_keys = range(len(states2plot))
    # print(env.obs_rms.mean, env.obs_rms.var)
    norm_obs = (states[states2plot].to_numpy() - env.obs_rms.mean[:6]) / np.sqrt(env.obs_rms.var[:6])
    norm_states = pd.DataFrame(data=norm_obs, columns=states2plot)

    print(norm_states["Air Temperature"].mean(), norm_states["Air Temperature"].var())

    # norm_obs = env.unnormalize_obs(states[states2plot].to_numpy())

    print(env.obs_rms.mean[:6], env.obs_rms.var[:6])

    fig, axes = createDistFig(states2plot)
    plotDistributions(fig, axes=axes, states=states, states2plot=states2plot, label=args.runname, color="C00")
    plotDistributions(fig, axes=axes, states=norm_states, states2plot=states2plot, label=args.runname, color="C01")
    plt.show()
