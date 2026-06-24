"""
van Hasselt et al.(2015)

environment : Ms PacMan
action space : MultiDiscrete([9 9])

"""
import gymnasium as gym 
import ale_py
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.wrappers.transform_observation import GrayscaleObservation,ResizeObservation

import numpy as np
import torch,sys,random,mlflow
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.distributions import Categorical
from torch.optim import Adam

from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from itertools import chain
from tqdm import tqdm


MAX_EP_STEPS = 500
NUM_ENVS = 20
R_SHAPE = (150,150)
# -
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 2_000 
GAMMA = .99
LR = None 

# TODO : change env to frozen lake
def vec_env():
    def make():
        x = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS)
        x = GrayscaleObservation(x)
        x = ResizeObservation(x,R_SHAPE)
        # TODO : frame stack,skip,obs reshape
        return x # [1, 150, 150]
    return AsyncVectorEnv([make for _ in range(NUM_ENVS)])


# TODO : remove gradient method
class q_function(nn.Module):
    def __init__(self):  
        super().__init__()
        self.c1 = nn.LazyConv2d( 64,1,1)
        self.c2 = nn.LazyConv2d(128,3,2)
        self.c3 = nn.LazyConv2d(128,3,2)
        self.c4 = nn.LazyConv2d(128,3,2)

        self.l1 = nn.LazyLinear(2048)
        self.l2 = nn.LazyLinear(1024)
        self.l3 = nn.LazyLinear(512)
        self.l4 = nn.LazyLinear(9)

    def forward(self,s):
        x = F.silu(self.c1(s)) # B,64, 150, 150
        x = F.silu(self.c2(x)) # B,128, 74, 74
        x = F.silu(self.c3(x)) # B,128, 36, 36
        x = F.silu(self.c4(x)) # B,128, 17, 17

        x = F.silu(self.l1(x.flatten(1)))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        x = self.l4(x)
        return F.softmax(x,-1) 


class ddqn:
    def __init__(self,start = False,storage_path=None):
        self.start = start
        self.env = vec_env()
  
        q_function()(torch.randint(0,1255,(NUM_ENVS,1,150,150),dtype=torch.float,device=DEVICE))
        self.q1 = q_function()
        self.q2 = deepcopy(self.q1)
        self.buffer = deque(maxlen=MAX_EP_STEPS)

        self.q1.to(DEVICE) ; self.q2.to(DEVICE) 
        self.q1.compile()  ; self.q2.compile()

        self.optim = Adam(chain(self.q1.parameters(),self.q2.parameters()),lr=LR)
    
    # TODO remove save function
    def save(self,storage_path):
        data = {
            "q1 state":self.q1.state_dict(),
            "q2 state":self.q2.state_dict(),
            "optim state":self.optim.state_dict()
        }
        torch.save(data,f"{storage_path}/state_{n}.pth")
    
    def q_update(self,transition,q1,q2):
        with torch.no_grad():
            state,nx_state,reward,done,action = transition
            
            # TODO : fix formula
            argmax = torch.argmax(q1(nx_state),1)
            disc_q2 = GAMMA * q2(nx_state) # discounted q2 value
            target = (reward + disc_q2.mean(-1) * (1-done.float())).unsqueeze(-1) 
        
        new_q = q1(state)
        new_q = torch.gather(new_q,1,action.unsqueeze(-1))

        loss = F.smooth_l1_loss(new_q,target).mean()
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        return loss.item()

    def run(self):
        if self.start:
            with mlflow.start_run() as run:
                self.state = tensor(self.env.reset()[0],dtype=torch.float,device=DEVICE).unsqueeze(1)
                
                for n in tqdm(range(MAX_STEPS),total=MAX_STEPS):
                    #TODO : FIX LOOP LOGIC AND ADD EGREEDY
                    for i in range(MAX_EP_STEPS):
                        with torch.no_grad():
                            action = torch.argmax(self.q1(self.state) + self.q2(self.state),dim=1)
                            nx_state,reward,done,trunc,info = self.env.step(action.long().tolist())
                            self.buffer.append(
                                [
                                    self.state,
                                    tensor(nx_state,dtype=torch.float,device=DEVICE).unsqueeze(1),
                                    tensor(reward,device=DEVICE),
                                    tensor(done,device=DEVICE),
                                    action
                                ]
                            )
                            self.state = tensor(nx_state,dtype=torch.float,device=DEVICE).unsqueeze(1) 
                            mlflow.log_metrics({"reward":reward.mean().item()},step=n)
                        
                        #TODO MOVE TO SOFT UPDATE
                        """
                        for transition in self.buffer:
                            if
                                loss = self.q_update(transition,self.q1,self.q2)
                            else:
                                loss = self.q_update(transition,self.q2,self.q1)
                            mlflow.log_metrics({"loss":loss},step=n)
                        """
                if n%100== 0:
                    self.save(n) # state dicts saving


