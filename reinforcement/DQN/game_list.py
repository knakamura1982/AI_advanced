import gymnasium as gym
from gymnasium import envs


for name in envs.registry.keys():
    print(name)
