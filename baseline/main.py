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


MAX_EP_STEPS = 500
NUM_ENVS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vec_env():
    def make():
        x = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS)
        # TODO
        # grayscale
        # frame stack,skip,obs reshape
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

    def forward(self,s,a): # q(s,a) -> q value
        return None


class buffer:
    def __init__(self,env=None,policy=None,q_function=None):
        self.env = env
        self.policy = policy
        self.q_functin = q_function
        
        self.b_q_values = torch.zeros(NUM_ENVS,1,dtype=torch.half,device=DEVICE)
        self.b_q_target = self.b_q_values.clone().detach()
        self.b_cur_states = torch.zeros(NUM_ENVS,210,160,3,dtype=torch.half,device=DEVICE) # TODO : squeeze -1 dim
        self.b_nx_states = self.b_curr_state.clone().detach()
        self.b_reward = torch.zeros(NUM_ENVS,1,dtype=torch.half,device=DEVICE) # TODO unsqueeze -1 dim
        self.done = torch.zeros(NUM_ENVS,1,dtype=torch.bool,device=DEVICE) # ''

        self.step_num = 0

    def step(self):
        self.step_num+=1
        
        self.env.reset()
        action = self.env.action_space.sample()
        state,reward,done,trunc,info = self.env.step(action)
        
        """
        self.b_cur_states[self.step_num].copy_(torch.from_numpy())
        self.b_nx_states[self.step_num].copy_(torch.from_numpy(states))
        self.b_reward[self.step_num].copy_(torch.from_numpy(reward))
        self.b_done[self.step_num].copy_(torch.from_numpy(done))
        """

        """
        if self.step_num % MAX_EP_STEPS:
            # TODO : compute target
            pass
        """
           
    def sample(self,batch):
        pass


class ddqn:
    def __init__(start=False,storage_path=None):
        self.env = vec_env()
        self.policy = policy() ; self.policy.to(DEVICE) ; # self.policy.compile()
        self.q_func = q_func() ; self.q_func.to(DEVICE) ; # self.q_func.compile()
    
    def save(self,storage_path):
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
    #ddqn().run(True,storage_path="./")

    
