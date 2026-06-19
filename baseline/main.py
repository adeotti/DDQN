import gymnasium as gym 
import ale_py
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.wrappers.transform_observation import GrayscaleObservation,ResizeObservation

import numpy as np
import torch,sys
import torch.nn as nn
from torch.optim import adam

from copy import deepcopy
from dataclasses import dataclass
import mlflow


# -
MAX_EP_STEPS = 500
NUM_ENVS = 2
R_SHAPE = (150,150)
# -
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = int(1e4) # int(1e6) # same as in the original paper
GAMMA = .99
BATCH = 512
LR = int(1e-4)


def vec_env():
    def make():
        x = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS)
        x = GrayscaleObservation(x)
        x = ResizeObservation(x,R_SHAPE)
        # TODO : frame stack,skip,obs reshape
        return x
    return AsyncVectorEnv([make for _ in range(NUM_ENVS)])


class q_function(nn.Module):
    def __init__(self):
        pass

    def forward(self,s,a): # q(s,a) -> q value
        return None


class buffer:
    def __init__(self,env=None,q_function=None):
        self.env = env
        self.state = torch.as_tensor(self.env.reset()[0],device=DEVICE) # state s0
        self.q_function = q_function
        # - 
        self.b_q_values = torch.zeros(MAX_STEPS,NUM_ENVS,dtype=torch.half,device=DEVICE)
        self.b_q_target = self.b_q_values.clone().detach()
        self.b_curr_states = torch.zeros(MAX_STEPS,NUM_ENVS,*R_SHAPE,dtype=torch.half,device=DEVICE) 
        self.b_nx_states = self.b_curr_states.clone().detach()
        self.b_reward = torch.zeros(MAX_STEPS,NUM_ENVS,dtype=torch.half,device=DEVICE)
        self.b_done = torch.zeros(MAX_STEPS,NUM_ENVS,dtype=torch.bool,device=DEVICE) 
        #-
        self.step_num = 0

    def step(self):
        with torch.no_grad():
            self.step_num+=1
            
            # q val = self.q_function(self.state,action)
            action = self.env.action_space.sample()
            nx_state,reward,done,trunc,info = self.env.step(action) 

            self.b_curr_states[self.step_num].copy_(self.state)
            self.b_nx_states[self.step_num].copy_(torch.from_numpy(nx_state))
            self.b_reward[self.step_num].copy_(torch.from_numpy(reward))
            self.b_done[self.step_num].copy_(torch.from_numpy(done))
         
            # TD(0) 
            target = torch.as_tensor(reward) + GAMMA # * max(self.q_function(states,action))
            self.b_q_target[self.step_num].copy_(target)

            self.state = torch.as_tensor(nx_state,device=DEVICE)
            
            return torch.from_numpy(reward)

    def sample(self,batch):
        pass


class ddqn:
    def __init__(start=False,storage_path=None):
        self.env = vec_env()
        self.q_func = q_func()    
        self.q_targ = deep_copy(self.q_function)
        self.q_func.to(DEVICE) ; self.q_targ.to(DEVICE) 
        #self.q_func.compile() ; self.q_targ.compile()

        self.optim = adam(self.q_func.parameters(),lr=LR)
    
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
    r = buffer(vec_env()).step()
    print(r)
   
    
