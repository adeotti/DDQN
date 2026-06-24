# environment : Ms PacMan
# action space : MultiDiscrete([9 9])

import gymnasium as gym 
import ale_py
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.wrappers.transform_observation import GrayscaleObservation,ResizeObservation

import numpy as np
import torch,sys,random
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from itertools import chain
from tqdm import tqdm
import mlflow


MAX_EP_STEPS = 5 # 500
NUM_ENVS = 2
R_SHAPE = (150,150)
# -
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 10 # int(1e6) 
GAMMA = .99
LR = int(1e-4)


def vec_env():
    def make():
        x = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS)
        x = GrayscaleObservation(x)
        x = ResizeObservation(x,R_SHAPE)
        # TODO : frame stack,skip,obs reshape
        return x # [1, 150, 150]
    return AsyncVectorEnv([make for _ in range(NUM_ENVS)])


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
        #self.q1.compile()  ; self.q2.compile()

        self.optim = Adam(chain(self.q1.parameters(),self.q2.parameters()),lr=LR)
    
    def save(self,storage_path):
        pass
    
    def q_update(self,transition,q1,q2):
        with torch.no_grad():
            state,nx_state,reward,done,action = transition
           
            argmax = torch.argmax(q1(nx_state),1)
            disc_q2 = GAMMA * q2(nx_state) # discounted q2 value
            target = reward + disc_q2.mean(-1) * (1-done.float())
        
        new_q = q1(state)
      
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
                        
                        for transition in self.buffer:
                            if random.random() < 0.5:
                                loss = self.q_update(transition,self.q1,self.q2)
                            else:
                                loss = self.q_update(transition,self.q2,self.q1)
                            mlflow.log_metrics({"loss":loss},step=n)
                        
                            #sys.exit("here")

                    
if __name__ == "__main__":
    ddqn("./").run()
      
    
