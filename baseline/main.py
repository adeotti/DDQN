# Baseline environment: Ms. Pac-Man (Atari)
# Results in Figure S4, page 21 of the original paper
# Reward should peak around 250k steps

import gymnasium as gym 
import ale_py
from gymnasium.vector.async_vector_env import AsyncVectorEnv
import numpy as np
import torch,sys
import torch.nn as nn
from torch.optim import adam
from copy import deepcopy
from dataclasses import dataclass
import mlflow


# - hypers 
NUM_ENVS = 2

def vec_env():
    def make():
        x = gym.make("ALE/MsPacman-v5")
        # - norm etc
        return x
    return AsyncVectorEnv([make for _ in range(NUM_ENVS)])


class policy(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(s): # p(s) -> a
        return None


class q_function(nn.Module):
    def __init__(self):
        pass

    def forward(self): # q(s,a) -> q value
        return None


class buffer:
    def __init__(self):
        pass

    def step(self):
        pass

    def sample(self,batch):
        pass

class ddqn:
    def __init__(start=False):
        pass
    
    def save(self):
        pass
    
    def run(self,storage_path=None):
        if start:
            with mlflow.start_run() as run:
                
                ...

                mlflow.log_metrics(
                    {
                    
                        },
                    step = None
                )
        

if __name__ == "__main__":
    ddqn().run(True,storage_path="./")
    
