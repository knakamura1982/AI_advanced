import gym
from gym import envs


for spec in envs.registry.all():
    print(spec.id)
