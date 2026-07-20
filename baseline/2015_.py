"""
Multi-threaded DDQN version of the 2015.py file 
"""
import gymnasium as gym 
import ale_py
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.wrappers.transform_observation import GrayscaleObservation,ResizeObservation
from gymnasium.wrappers import FrameStackObservation

import torch,sys,random,mlflow,time
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from collections import deque
from tqdm import tqdm
from threading import Thread,Event
from queue import Queue


MAX_EP_STEPS = 500
NUM_ENVS = 10
R_SHAPE = (100,100)
# -
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = int(35e5)
GAMMA = .99
LR = 25e-5
BATCH_SIZE = 32
Q1_NET_UPDATE_FREQ = MAX_STEPS // 4 # update q1 weights every 4 steps 
TARGET_NET_UPDATE_FREQ = int(10e3)  # update q target weights every 10k steps
BUFFER_SIZE = 100_000 // NUM_ENVS


def vec_env():
    def make():
        x = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS)
        x = GrayscaleObservation(x)
        x = ResizeObservation(x,R_SHAPE)
        x = FrameStackObservation(x,4)
        return x 
    return AsyncVectorEnv([make for _ in range(NUM_ENVS)])


class q_function(nn.Module):
    def __init__(self):  
        super().__init__()
        self.c1 = nn.LazyConv2d(32,8,4)  
        self.c2 = nn.LazyConv2d(64,4,2)
        self.c3 = nn.LazyConv2d(64,3,1)

        self.l1 = nn.LazyLinear(512)
        self.l2 = nn.LazyLinear(9)

    def forward(self,s):
        x = s / 255.
        x = self.c1(x)
        x = F.silu(self.c2(x))
        x = F.silu(self.c3(x))
        
        x = F.silu(self.l1(x.flatten(1)))
        return self.l2(x)


class ddqn:
    def __init__(self,storage_path="./"):
        self.storage_path = storage_path
        self.env = vec_env()
        self.channels = self.env.observation_space.shape[1]
        
        dummy_obs = (torch.randint(0,255,(NUM_ENVS,self.channels,*R_SHAPE),dtype=torch.float))
        self.q1 = q_function()
        self.q1(dummy_obs)
        self.q1 = q_function().to(DEVICE)
        self.q1.compile(mode="max-autotune")
        
        self.target_net = deepcopy(self.q1).to(DEVICE)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optim = torch.optim.Adam(self.q1.parameters(),lr=LR,fused=True)
        self.reward_data = torch.zeros(NUM_ENVS, dtype=torch.float)
        self.global_step = 0
        self.target_sync_event = Event() # tracking global steps

        self.episode_queue = Queue(maxsize=5) # holds full raw episodes
        self.batch_queue = Queue(maxsize=32)  # holds processed batches for GPU

        self.thread_1 = Thread(target=self.step_env,args=(self.episode_queue,),daemon=True)
        self.thread_2 = Thread(target=self.sample_processor,args=(self.episode_queue,self.batch_queue),daemon=True)


    def create_buffer(self):
        return (
            torch.zeros(MAX_EP_STEPS,NUM_ENVS,self.channels,*R_SHAPE),
            torch.zeros(MAX_EP_STEPS,NUM_ENVS,self.channels,*R_SHAPE),
            torch.zeros(MAX_EP_STEPS,NUM_ENVS),
            torch.zeros(MAX_EP_STEPS,NUM_ENVS),
            torch.zeros(MAX_EP_STEPS,NUM_ENVS,dtype=torch.int64)
        )


    @torch.no_grad()
    def step_env(self,queue):
        t_state,t_nx_state,t_reward,t_done,t_action = self.create_buffer()
        state = torch.tensor(self.env.reset()[0],dtype=torch.float,device="cpu")
        n = 0

        while True:
            decay_fraction = min(self.global_step / int(1e6), 1)
            epsilon = 1 - (1 - 0.1) * decay_fraction

            if random.random() < epsilon: action = self.env.action_space.sample()
            else: action = torch.argmax(self.q1(state.to(DEVICE)), dim=1).tolist()

            nx_state, reward, done, trunc, _ = self.env.step(action)
            self.reward_data += reward

            t_state[n].copy_(state)
            t_nx_state[n].copy_(torch.tensor(nx_state))
            t_reward[n].copy_(torch.tensor(reward))
            t_done[n].copy_(torch.tensor(done))
            t_action[n].copy_(torch.tensor(action))

            n += 1
            self.global_step += 1
            if self.global_step > 0 and self.global_step % TARGET_NET_UPDATE_FREQ == 0:
                self.target_sync_event.set()

            if n == MAX_EP_STEPS:
                queue.put((t_state.clone(),t_nx_state.clone(),t_reward.clone(),t_done.clone(),t_action.clone()))
                t_state,t_nx_state,t_reward,t_done,t_action = self.create_buffer()
                n = 0

            state = torch.tensor(nx_state,dtype=torch.float,device="cpu")


    def sample_processor(self,ep_queue,batch_queue):
        b_state = torch.zeros(BUFFER_SIZE,NUM_ENVS,self.channels,*R_SHAPE,dtype=torch.uint8)
        b_nx_state = torch.zeros(BUFFER_SIZE,NUM_ENVS,self.channels,*R_SHAPE,dtype=torch.uint8)
        b_reward = torch.zeros(BUFFER_SIZE,NUM_ENVS)
        b_done = torch.zeros(BUFFER_SIZE,NUM_ENVS,dtype=torch.bool)
        b_action = torch.zeros(BUFFER_SIZE,NUM_ENVS,dtype=torch.int64)

        ptr = 0
        size = 0

        while True:
            ep_state,ep_nx_state,ep_reward,ep_done,ep_action = ep_queue.get()

            indices = torch.arange(ptr, ptr + MAX_EP_STEPS) % BUFFER_SIZE # circular buffer core indexing method 
            b_state[indices] = ep_state.to(torch.uint8)
            b_nx_state[indices] = ep_nx_state.to(torch.uint8)
            b_reward[indices] = ep_reward
            b_done[indices] = ep_done.to(torch.bool)
            b_action[indices] = ep_action

            ptr = (ptr + MAX_EP_STEPS) % BUFFER_SIZE
            size = min(size + MAX_EP_STEPS,BUFFER_SIZE)
        
            if size < 10_000:
                continue
            
            for _ in range(MAX_EP_STEPS//4): # sampling every 4 step, 500 steps -> 125 samples
                idx = torch.randint(0,size,(BATCH_SIZE,))
                s_state = b_state[idx].flatten(0,1).to(torch.float32)
                s_nx_state = b_nx_state[idx].flatten(0,1).to(torch.float32)
                s_reward = b_reward[idx].reshape(-1)
                s_done = b_done[idx].reshape(-1).to(torch.float32)
                s_action = b_action[idx].reshape(-1,1)

                batch_queue.put((s_state,s_nx_state,s_reward,s_done,s_action))
    
    
    def save(self,n): 
        data = { "q1 state":self.q1.state_dict(),
            "target net state":self.target_net.state_dict(),
            "optim state":self.optim.state_dict()
        }
        torch.save(data,f"{self.storage_path}/state_{n}.pth")


    @torch.compile(mode="max-autotune")
    def compute_loss(self,s_nx_state,s_reward,s_done,pred_q):
        with torch.no_grad():
            nx_action = torch.argmax(self.q1(s_nx_state),dim=1).unsqueeze(-1)
            eval_ = self.target_net(s_nx_state).gather(1,nx_action).squeeze(1)
            target = s_reward + GAMMA * eval_ * (1 - s_done)
        return F.mse_loss(pred_q,target)
    

    def main(self):
        self.thread_1.start()
        self.thread_2.start()

        mlflow.set_experiment("pacman")
        with mlflow.start_run() as run:

            while self.batch_queue.qsize() < 10:
                time.sleep(0.1)

            for t in tqdm(range((MAX_STEPS//MAX_EP_STEPS) + 1),total=(MAX_STEPS//MAX_EP_STEPS) + 1):
                s_state,s_nx_state,s_reward,s_done,s_action = self.batch_queue.get()
                
                s_state = s_state.to(DEVICE,non_blocking=True)
                s_nx_state = s_nx_state.to(DEVICE,non_blocking=True)
                s_reward = s_reward.to(DEVICE,non_blocking=True)
                s_done = s_done.to(DEVICE,non_blocking=True)
                s_action = s_action.to(DEVICE,non_blocking=True)

                with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16):
                    pred_q = self.q1(s_state).gather(1,s_action).squeeze(1)
                    loss = self.compute_loss(s_nx_state,s_reward,s_done,pred_q)
                    current_loss = loss.item()

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                self.optim.step()

                if self.target_sync_event.is_set():
                    self.target_net.load_state_dict(self.q1.state_dict())
                    self.target_sync_event.clear() # reset the flag !!

                if t > 0 and t % 500 == 0:
                    self.save(t)

                    mlflow.log_metrics(
                        {
                        "average reward": self.reward_data.mean().item(),
                        "loss": current_loss,
                        },
                    step=self.global_step
                    ) 
                    self.reward_data = torch.zeros(NUM_ENVS,dtype=torch.float) 

    
    def test(self):
        env = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS,render_mode="human")
        env = GrayscaleObservation(env)
        env = ResizeObservation(env,R_SHAPE)
        env = FrameStackObservation(env,4)
        state = env.reset()[0]
        
        policy = q_function()
        checkpoint = torch.load("./state_7000.pth", map_location=torch.device("cpu"))
        compiled_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["q1 state"].items()}
        policy.load_state_dict(compiled_state_dict)

        for n in range(500*10):
            state = torch.tensor(state,dtype=torch.float).unsqueeze(0) 
            nx_s,reward,done,trunc,_ = env.step(torch.argmax(policy(state)).item())
            state = nx_s
           
            env.render()
            if done or trunc:
                break


if __name__ == "__main__": 
    import warnings,logging
    warnings.filterwarnings("ignore") ; logging.disable(logging.CRITICAL)
    ddqn("./").test()
    
