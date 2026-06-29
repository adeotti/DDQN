"""
van Hasselt et al.(2015)

environment : Ms PacMan
https://ale.farama.org/environments/ms_pacman/
"""
import gymnasium as gym 
import ale_py
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.wrappers.transform_observation import GrayscaleObservation,ResizeObservation

import torch,sys,random,mlflow
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from collections import deque
from itertools import chain
from tqdm import tqdm


MAX_EP_STEPS = 500
NUM_ENVS = 2#10
R_SHAPE = (100,100)
# -
EPSILON = 1 # todo : linear decay until 1 M and static until end of training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 2_000 
GAMMA = .99
LR = 25e-5
TAU = 0.5
BATCH_SIZE = 32
Q1_NET_UPDATE_FREQ = MAX_EP_STEPS / 4 # update very four steps 
TARGET_NET_UPDATE_FREQ = int(10e3)


def vec_env():
    def make():
        x = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS)
        x = GrayscaleObservation(x)
        x = ResizeObservation(x,R_SHAPE) 
        return x # [1, 150, 150]
    return AsyncVectorEnv([make for _ in range(NUM_ENVS)])


class q_function(nn.Module):
    def __init__(self):  
        super().__init__()
        self.c1 = nn.LazyConv2d( 32,1,1)
        self.c2 = nn.LazyConv2d(64,3,2)
        self.c3 = nn.LazyConv2d(64,3,2)

        self.l1 = nn.LazyLinear(1024)
        self.l2 = nn.LazyLinear(512)
        self.l3 = nn.LazyLinear(9)

    def forward(self,s):
        x = F.silu(self.c1(s)) 
        x = F.silu(self.c2(x)) 
        x = F.silu(self.c3(x)) 

        x = F.silu(self.l1(x.flatten(1)))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        return x


class ddqn:
    def __init__(self,storage_path=None):
        self.storage_path = storage_path
        self.env = vec_env()
  
        q_function()(torch.randint(0,255,(NUM_ENVS,1,*R_SHAPE),dtype=torch.float)) # init
        self.q1 = q_function()
        self.target_net = deepcopy(self.q1)
        self.buffer = deque(maxlen=MAX_EP_STEPS)

        self.q1.to(DEVICE) 
        self.target_net.to(DEVICE) 
        #self.q1.compile()
        #self.target_net.compile()

        self.optim = torch.optim.Adam(chain(self.q1.parameters(),self.target_net.parameters()),lr=LR)
        self.to_tensor = lambda x : torch.tensor(x,dtype=torch.float,device=DEVICE)
        self.reward_data = torch.zeros(NUM_ENVS,dtype=torch.float)
        self.step_count = 0
    
    def save(self,n):
        data = {
            "q1 state":self.q1.state_dict(),
            "target net state":self.target_net.state_dict(),
            "optim state":self.optim.state_dict()
        }
        torch.save(data,f"{self.storage_path}/state_{n}.pth")
    
    def main(self):
        with mlflow.start_run() as run:

            self.state = torch.tensor(self.env.reset()[0],dtype=torch.float,device=DEVICE).unsqueeze(1)
            for n in tqdm(range(MAX_STEPS),total=MAX_STEPS):

                for i in range(MAX_EP_STEPS):
                    with torch.no_grad():
                        if random.random() < EPSILON : action = self.env.action_space.sample()
                        else : action = torch.argmax(self.q1(self.state),dim=1).tolist()
                        
                        nx_state,reward,done,trunc,_ = self.env.step(action)
                        self.reward_data += reward
                        
                        data = [self.state,nx_state,reward,done,action] 
                        data = list(map(self.to_tensor,data))
                        self.buffer.append(data)
                        
                        self.state = torch.tensor(nx_state,dtype=torch.float,device=DEVICE).unsqueeze(1)
                        self.step_count += 1
            
                for t in range(Q1_NET_UPDATE_FREQ):# update every 4 steps
                    # TODO : batch items
                    id_ = random.randint(0,500-1)
                    s,nx,r,d,a = self.buffer[id_] # s : state, nx : state t+1, r : reward, d : done, a : action
                
                    pred_q = self.q1(s).gather(1,a.unsqueeze(0).long()).squeeze()
                
                    with torch.no_grad(): # target q computation
                        # prediciton using q1 -> eval of q1 prediction using Q target -> TD(0) 
                        nx_action = torch.argmax(self.q1(nx.unsqueeze(1)),1).unsqueeze(0)
                        eval_ = self.target_net(nx.unsqueeze(1)).gather(1,nx_action).squeeze()
                        target = r + GAMMA * eval_ * (1-d)
            
                    loss = F.mse_loss(pred_q,target).mean()
          
                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optim.step()
                    
                if self.step_count % TARGET_NET_UPDATE_FREQ == 0: # update target net every 10k steps
                    self.q_target.load_state_dict(self.q1.state_dict())
                        
                if n%100 == 0:
                    self.save(n) 
                    mlflow.log_metrics({
                        "average reward":self.reward_data.mean().item(),
                        "loss":loss.item(),
                    },step=n)

    
    def test(self):
        env = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS,render_mode="human")
        env = GrayscaleObservation(env)
        env = ResizeObservation(env,R_SHAPE)
        state = env.reset()
        
        policy = q_function()
        policy.load_state_dict(torch.load("./")["q1 state"])

        for n in range(500*10):
            nx_s,reward,done,trunc,_ = env.step(policy(torch.tensor(state),dtype=torch.float).item())
            state = nx_state
           
            env.render()
            if done or trunc:
                break



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    ddqn("./").main()
