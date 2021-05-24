"""

Utility script to train diff gym compatible libraries with ppo

"""
import argparse
import yaml
from ppo import PPO

parser = argparse.ArgumentParser()
parser.add_argument("-e","--exp",type=str, required=True,help="The experiment name as defined in the yaml file")

with open("./experiments.yaml") as f:
    experiments = yaml.safe_load(f)

args = parser.parse_args()

print(args)

experiment = args.exp
hyperparams = experiments[experiment]

algo = PPO(device="cpu",**hyperparams)

algo.learn()

