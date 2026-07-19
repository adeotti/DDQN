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
from tqdm import tqdm


MAX_EP_STEPS = 500
NUM_ENVS = 10
R_SHAPE = (100,100)
# -
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 7_000 
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
        return x # [4, 150, 150]
    return AsyncVectorEnv([make for _ in range(NUM_ENVS)])


env = gym.make("ALE/MsPacman-v5")
env = GrayscaleObservation(env)
env = ResizeObservation(env,R_SHAPE)


class q_function(nn.Module):
    def __init__(self):  
        super().__init__()
        self.c1 = nn.LazyConv2d(32,8,4)  
        self.c2 = nn.LazyConv2d(64,4,2)
        self.c3 = nn.LazyConv2d(64,3,1)

        self.l1 = nn.LazyLinear(512)
        self.l2 = nn.LazyLinear(9)

    def forward(self,s):
        x = self.c1(s)
        x = F.silu(self.c2(x))
        x = F.silu(self.c3(x))
        
        x = F.silu(self.l1(x.flatten(1)))
        return self.l2(x)


class ddqn:
    def __init__(self,storage_path=None):
        self.storage_path = storage_path
        self.env = vec_env()
        self.channels = self.env.observation_space.shape[1]
    
        dummy_obs = (torch.randint(0,255,(NUM_ENVS,self.channels,*R_SHAPE),dtype=torch.float))
        self.q1 = q_function()
        self.q1(dummy_obs)
        self.target_net = deepcopy(self.q1)

        self.q1.to(DEVICE) 
        self.target_net.to(DEVICE)

        self.q1.compile(mode="max-autotune-no-cudagraphs")#(mode="max-autotune") 
        self.target_net.compile(mode="max-autotune-no-cudagraphs")#(mode="max-autotune")

        self.optim = torch.optim.Adam(self.q1.parameters(),lr=LR)
        self.reward_data = torch.zeros(NUM_ENVS,dtype=torch.float)
        self.init_storage()
   

    def init_storage(self):
        b = MAX_EP_STEPS 
        self.t_state = torch.zeros(b,NUM_ENVS,self.channels,*R_SHAPE)
        self.t_nx_state = torch.zeros(b,NUM_ENVS,self.channels,*R_SHAPE)
        self.t_reward = torch.zeros(b,NUM_ENVS)
        self.t_done = torch.zeros(b,NUM_ENVS)
        self.t_action = torch.zeros(b,NUM_ENVS,dtype=torch.int64)

        self.t_state = self.t_state.pin_memory()
        self.t_nx_state = self.t_nx_state.pin_memory()
        self.t_reward = self.t_reward.pin_memory()
        self.t_done = self.t_done.pin_memory()
        self.t_action = self.t_action.pin_memory()


    def save(self,n): 
        data = { "q1 state":self.q1.state_dict(),
            "target net state":self.target_net.state_dict(),
            "optim state":self.optim.state_dict()
        }
        torch.save(data,f"{self.storage_path}/state_{n}.pth")


    @torch.compile(mode="max-autotune-no-cudagraphs")#(mode="max-autotune")
    def compute_loss(self,s_nx_state,s_reward,s_done,pred_q):
        with torch.no_grad(): # target q
            # prediciton using q1 -> eval of q1 prediction using Q target -> TD(0) 
            nx_action = torch.argmax(self.q1(s_nx_state),1).unsqueeze(-1)
            eval_ = self.target_net(s_nx_state).gather(1,nx_action) 
            target = s_reward + GAMMA * eval_ * (1-s_done)
            
        loss = F.mse_loss(pred_q,target).mean()
        return loss

    def main(self,gpu_strean=None):
        mlflow.set_experiment("pacman")
        with mlflow.start_run() as run:

            self.state = torch.as_tensor(self.env.reset()[0],dtype=torch.float)
            gpu_state = torch.zeros_like(self.state,device=DEVICE)
            global_step = 0

            for n in tqdm(range(MAX_STEPS),total=MAX_STEPS):
                
                for i in range(MAX_EP_STEPS):
                    with torch.no_grad():
                        gpu_state.copy_(self.state)

                        global_step += 1

                        decay_fraction = min(global_step/int(1e6),1) # linearly decay epsilon from 1 to 0.1 over 1M steps
                        epsilon = 1 - (1 - 0.1) * decay_fraction

                        if random.random() < epsilon : action = self.env.action_space.sample()
                        else : action = torch.argmax(self.q1(gpu_state),dim=1).tolist()

                        nx_state,reward,done,trunc,_ = self.env.step(action)
                        self.reward_data += reward
            
                        self.t_state[i].copy_(self.state)
                        self.t_nx_state[i].copy_(torch.as_tensor(nx_state))
                        self.t_reward[i].copy_(torch.as_tensor(reward))
                        self.t_done[i].copy_(torch.as_tensor(done))
                        self.t_action[i].copy_(torch.as_tensor(action))

                        self.state = torch.as_tensor(nx_state,dtype=torch.float)

                for t in range(MAX_EP_STEPS//4):
                    idx = torch.randint(0,MAX_EP_STEPS,(BATCH_SIZE,))
                    s_state = self.t_state[idx].flatten(0,1).to(DEVICE,non_blocking=True)
                    s_nx_state = self.t_nx_state[idx].flatten(0,1).to(DEVICE,non_blocking=True)
                    s_reward = self.t_reward[idx].reshape(-1).to(DEVICE,non_blocking=True)  # (steps*envs,)
                    s_done = self.t_done[idx].reshape(-1).to(DEVICE,non_blocking=True)  # (steps*envs,)
                    s_action = self.t_action[idx].reshape(-1,1).to(DEVICE,non_blocking=True) # (steps*envs,1)
                    
                    with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16):
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

                    current_loss = loss.item()

                if n > 0 and n % TARGET_NET_UPDATE_FREQ == 0: # hard update of target net every 10k steps
                    target_net.load_state_dict(q1.state_dict())

                if n > 0 and n % 500 == 0:
                    self.save(n) 
                    mlflow.log_metrics({
                        "average reward":self.reward_data.mean().item(),
                        "loss":current_loss,
                    },step=n)

                    self.reward_data = torch.zeros(NUM_ENVS,dtype=torch.float)


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
    
