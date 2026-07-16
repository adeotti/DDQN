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

import torch,sys,random,mlflow,os
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
WARMUP_STEPS = 100
MAX_STEPS = int(35e5)
GAMMA = .99
LR = 25e-5
BATCH_SIZE = 32
Q1_NET_UPDATE_FREQ = MAX_STEPS // 4 # update q1 weights every 4 steps 
TARGET_NET_UPDATE_FREQ = int(10e3)  # update q target weights every 10k steps


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
    
        x = F.relu(self.l1(x.flatten(1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
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

        # thread 1 -> step env and collecting transitions
        self.queue = Queue(maxsize=10000)
        self.thread_1 = Thread(target=self.step_env,args=(self.queue,),daemon=False)
        self.thread_1.start()

        # thread 2 -> pulling the Queue and stacking data for the GPU
        self.thread_2 = Thread(target=self.sample,args=(self.queue,),daemon=False)
        self.gpu_stream = Queue(maxsize=10000)


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


    @torch.no_grad()
    def step_env(self,queue):
        data_list = []
        for n in range(MAX_STEPS):
            if self.step_count == 0:
                self.state = torch.tensor(self.env.reset()[0],dtype=torch.float,device="cpu")

            decay_fraction = min(self.step_count/int(1e6),1) # Linearly decay epsilon from 1 to 0.1 over 1M steps
            epsilon = 1 - (1 - 0.1) * decay_fraction

            if random.random() < epsilon : 
                action = self.env.action_space.sample()
            else :
                action = torch.argmax(self.q1(self.state.to(DEVICE)),dim=1).tolist()
                        
            nx_state,reward,done,trunc,_ = self.env.step(action)
            self.reward_data += reward
            
            data_list.append((self.state,nx_state,reward,done,action))
            if n > 0  and n % MAX_EP_STEPS == 0 :
                queue.put((data_list))

                if not self.thread_2.is_alive(): # start thread 2 after the first episode data collection
                    self.thread_2.start()
                    
                data_list = []
                                                
            self.state = torch.tensor(nx_state,dtype=torch.float,device="cpu")
            self.step_count += 1


    def sample(self,queue):
        while True:
            episode = queue.get() # episode should be lenght MAX_EP_STEPS
            
            # s : state , nx : next state , r : reward , d : done , a : actions
            s = [item[0] for item in episode]
            nx = [item[1] for item in episode]
            r = [item[2] for item in episode]
            d = [item[3] for item in episode]
            a = [item[4] for item in episode]
            
            # batching
            b_state = self.stack__(s)
            b_state = b_state.reshape(-1,1,*R_SHAPE)  # (steps*envs,1,R_SHAPE)

            b_nx_state = self.stack__(nx)
            b_nx_state = b_nx_state.unsqueeze(2).reshape(-1,1,*R_SHAPE) # (steps*envs,1,R_SHAPE)

            b_reward = self.stack__(r).reshape(-1)    # (steps*envs,)
            b_done   = self.stack__(d).reshape(-1)    # (steps*envs,)
            b_action = self.stack__(a).reshape(-1, 1) # (steps*envs,1)
            
            # random sampling
            idx = torch.randperm(BATCH_SIZE)
            s_state = b_state[idx].to(DEVICE)
            s_nx_state = b_nx_state[idx].to(DEVICE)
            s_reward = b_reward[idx].to(DEVICE)
            s_done = b_done[idx].to(DEVICE)
            s_action = b_action[idx].long().to(DEVICE)

            sample = (s_state,s_nx_state,s_reward,s_done,s_action)
            self.gpu_stream.put(sample)
            
    
    def main(self):
        mlflow.set_experiment("pacman")
        with mlflow.start_run() as run:

            if self.gpu_stream.qsize() > 10:

                for t in tqdm(range(Q1_NET_UPDATE_FREQ,total=Q1_NET_UPDATE_FREQ)):
                    s_state,s_nx_state,s_reward,s_done,s_action = self.gpu_stream.get() 
                                      
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
                    
                    if t > 0 and t % 2500 == 0: # update target net every 10k steps , 2500 = TARGET_NET_UPDATE_FREQ / 4
                        self.target_net.load_state_dict(self.q1.state_dict())
                        
                    if t > 0 and t % 1_000 == 0:
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
    
     






