"""
van Hasselt et al.(2015)

environment : Ms PacMan
https://ale.farama.org/environments/ms_pacman/
"""
import gymnasium as gym 
import ale_py
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.wrappers.transform_observation import GrayscaleObservation,ResizeObservation
from gymnasium.wrappers import FrameStackObservation

import torch,sys,random,mlflow
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from collections import deque
from itertools import chain
from tqdm import tqdm
from threading import Thread
from queue import Queue


MAX_EP_STEPS = 500
NUM_ENVS = 10
R_SHAPE = (100,100)
# -
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 7_000 
GAMMA = .99
LR = 25e-5
BATCH_SIZE = 32
Q1_NET_UPDATE_FREQ = MAX_EP_STEPS // 4 # update q1 weights every 4 steps 
TARGET_NET_UPDATE_FREQ = int(10e3)     # update q target weights every 10k steps


def vec_env():
    def make():
        x = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS)
        x = GrayscaleObservation(x)
        x = ResizeObservation(x,R_SHAPE)
        x = FrameStackObservation(x,4)
        return x # [1, 150, 150]
    return AsyncVectorEnv([make for _ in range(NUM_ENVS)])


env = gym.make("ALE/MsPacman-v5")
env = GrayscaleObservation(env)
env = ResizeObservation(env,R_SHAPE)

state = env.reset()[0]

class q_function(nn.Module):
    def __init__(self):  
        super().__init__()
        self.c1 = nn.LazyConv2d(32,1,1)
        self.c2 = nn.LazyConv2d(64,4,2)

        self.l1 = nn.LazyLinear(2048)
        self.l2 = nn.LazyLinear(1024)
        self.l3 = nn.LazyLinear(512)
        self.l4 = nn.LazyLinear(9)

    def forward(self,s):
        x = self.c1(s)
        x = F.relu(self.c2(x))
    
        x = F.silu(self.l1(x.flatten(1)))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        x = F.silu(self.l4(x))
        return x


class ddqn:
    def __init__(self,storage_path=None):
        self.storage_path = storage_path
        self.env = vec_env()
        channels = self.env.observation_space.shape[1]
  
        q_function()(torch.randint(0,255,(NUM_ENVS,channels,*R_SHAPE),dtype=torch.float))
        self.q1 = q_function()
        self.target_net = deepcopy(self.q1)
        self.buffer = deque(maxlen=MAX_EP_STEPS)

        self.q1.to(DEVICE) 
        self.target_net.to(DEVICE) 
        self.q1.compile() 
        self.target_net.compile()

        self.optim = torch.optim.Adam(chain(self.q1.parameters(),self.target_net.parameters()),lr=LR)
        self.reward_data = torch.zeros(NUM_ENVS,dtype=torch.float)
        self.step_count = 0
    
    def save(self,n): 
        data = { "q1 state":self.q1.state_dict(),
            "target net state":self.target_net.state_dict(),
            "optim state":self.optim.state_dict()
        }
        torch.save(data,f"{self.storage_path}/state_{n}.pth")


    def stack__(self,x):
        x = [torch.tensor(item, dtype=torch.float, device=DEVICE) for item in x]
        x = torch.stack(x)
        return x

    
    def main(self):
        mlflow.set_experiment("pacman")
        with mlflow.start_run() as run:

            self.state = torch.tensor(self.env.reset()[0],dtype=torch.float,device=DEVICE).unsqueeze(1)
            for n in tqdm(range(MAX_STEPS),total=MAX_STEPS):
                
                b_state,b_nx_state,b_reward,b_done,b_action = [],[],[],[],[]

                for i in range(MAX_EP_STEPS):
                    with torch.no_grad():
                        decay_fraction = min(n/int(1e6),1) # Linearly decay epsilon from 1 to 0.1 over 1M steps
                        epsilon = 1 - (1 - 0.1) * decay_fraction

                        if random.random() < epsilon : action = self.env.action_space.sample()
                        else : action = torch.argmax(self.q1(self.state),dim=1).tolist()
                        
                        nx_state,reward,done,trunc,_ = self.env.step(action)
                        self.reward_data += reward
                        
                        b_state.append(self.state)
                        b_nx_state.append(nx_state)
                        b_reward.append(reward) 
                        b_done.append(done) 
                        b_action.append(action)
                        
                        self.state = torch.tensor(nx_state,dtype=torch.float,device=DEVICE).unsqueeze(1)
                        self.step_count += 1
              
                b_state = self.stack__(b_state)
                b_state = b_state.reshape(-1,1,*R_SHAPE)  # (steps*envs,1,R_SHAPE)

                b_nx_state = self.stack__(b_nx_state)
                b_nx_state = b_nx_state.unsqueeze(2).reshape(-1,1,*R_SHAPE) # (steps*envs,1,R_SHAPE)

                b_reward = self.stack__(b_reward).reshape(-1)    # (steps*envs,)
                b_done   = self.stack__(b_done).reshape(-1)    # (steps*envs,)
                b_action = self.stack__(b_action).reshape(-1, 1) # (steps*envs,1)
                
                for t in range(Q1_NET_UPDATE_FREQ):
                    # sampling
                    idx = torch.randperm(BATCH_SIZE)
                    s_state = b_state[idx]
                    s_nx_state = b_nx_state[idx]
                    s_reward = b_reward[idx]
                    s_done = b_done[idx]
                    s_action = b_action[idx].long()
                    
                    pred_q = self.q1(s_state).gather(1,s_action).squeeze()
                
                    with torch.no_grad(): # target q
                        # prediciton using q1 -> eval of q1 prediction using Q target -> TD(0) 
                        nx_action = torch.argmax(self.q1(s_nx_state),1).unsqueeze(-1)
                        eval_ = self.target_net(s_nx_state).gather(1,nx_action) 
                        target = s_reward + GAMMA * eval_ * (1-s_done)
            
                    loss = F.mse_loss(pred_q,target).mean()
          
                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optim.step()
                    
                if self.step_count % TARGET_NET_UPDATE_FREQ == 0: # update target net every 10k steps
                    self.target_net.load_state_dict(self.q1.state_dict())
                        
                if n % 1_000 == 0:
                    self.save(n) 
                    mlflow.log_metrics({
                        "average reward":self.reward_data.mean().item(),
                        "loss":loss.item(),
                    },step=n)

    
    def test(self):
        env = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS,render_mode="human")
        env = GrayscaleObservation(env)
        env = ResizeObservation(env,R_SHAPE)
        state = env.reset()[0]
        
        q_function()(torch.randint(0,255,(1,1,*R_SHAPE),dtype=torch.float))
        policy = q_function()
        checkpoint = torch.load("./state_4000.pth", map_location=torch.device("cpu"))
        policy.load_state_dict(checkpoint["q1 state"])

        for n in range(500*10):
            state = torch.tensor(state,dtype=torch.float).unsqueeze(0).unsqueeze(0) 
            nx_s,reward,done,trunc,_ = env.step(torch.argmax(policy(state)).item())
            state = nx_s
           
            env.render()
            if done or trunc:
                break


if __name__ == "__main__": 
    import warnings,logging
    warnings.filterwarnings("ignore") ; logging.disable(logging.CRITICAL)
    ddqn("./").main()
    
     






