"""
Double Q-learning (Hasselt, 2010)
https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html
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

from collections import deque
from tqdm import tqdm


MAX_EP_STEPS = 500
NUM_ENVS = 20
R_SHAPE = (150,150)
MAX_STEPS = 2_000 
GAMMA = .95
#LR TODO use polynomial like in the paper 

def make_env():
    pass


class ddqn:
    def __init__(self,start = False,storage_path=None):
        self.start = start  
        self.buffer = deque(maxlen=MAX_EP_STEPS)

    def q_update(self,transition,q1,q2):
        with torch.no_grad():
            state,nx_state,reward,done,action = transition
            
            # TODO : fix formula
            argmax = torch.argmax(q1(nx_state),1)
            disc_q2 = GAMMA * q2(nx_state) # discounted q2 value
            target = (reward + disc_q2.mean(-1) * (1-done.float())).unsqueeze(-1) 
        
        new_q = q1(state)
        new_q = torch.gather(new_q,1,action.unsqueeze(-1))
        """
        loss = F.smooth_l1_loss(new_q,target).mean()
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        """
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
                                               
                        for transition in self.buffer:
                            if random.random() < 0.5:
                                loss = self.q_update(transition,self.q1,self.q2)
                            else:
                                loss = self.q_update(transition,self.q2,self.q1)
                                    
                    
if __name__ == "__main__":
    ddqn(False,"./").run() 
