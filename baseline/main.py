import gymnasium as gym 
from gymnasium.vector.async_vector_env import AsyncVectorEnv
import numpy as np
import torch,sys
import torch.nn as nn
from torch.optim import adam
from copy import deepcopy
from dataclasses import dataclass
import mlflow


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
    ddqn().run(False,storage_path="./")
        
