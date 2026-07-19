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
BUFFER_SIZE = 100_000


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
        x = self.c1(s)
        x = F.silu(self.c2(x))
        x = F.silu(self.c3(x))
        
        x = F.silu(self.l1(x.flatten(1)))
        return self.l2(x)


class ddqn:
    def __init__(self,storage_path="./"):
        self.storage_path = storage_path
        self.env = vec_env()
        self.channels = self.env.observation_space.shape[1]
        
        self.q1 = q_function().to(DEVICE)
        self.q1 = torch.compile(self.q1, mode="max-autotune")
        
        self.target_net = deepcopy(self.q1).to(DEVICE)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optim = torch.optim.Adam(self.q1.parameters(), lr=LR, fused=True)
        self.reward_data = torch.zeros(NUM_ENVS, dtype=torch.float)
        self.global_step = 0 

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

            if n == MAX_EP_STEPS:
                queue.put((t_state.clone(),t_nx_state.clone(),t_reward.clone(),t_done.clone(),t_action.clone()))
                t_state,t_nx_state,t_reward,t_done,t_action = self.create_buffer()
                n = 0

            state = torch.tensor(nx_state,dtype=torch.float,device="cpu")


    def sample_processor(self, ep_queue, batch_queue):
        replay_buffer = deque(maxsize=BUFFER_SIZE)

        while True:
            ep_state,ep_nx_state,ep_reward,ep_done,ep_action = ep_queue.get()
            
            for _ in range(MAX_EP_STEPS//4): # sampling every 4 step, 500 steps -> 125 samples
                idx = torch.randint(0, MAX_EP_STEPS, (BATCH_SIZE,))
                s_state = ep_state[idx].flatten(0,1)
                s_nx_state = ep_nx_state[idx].flatten(0,1)
                s_reward = ep_reward[idx].reshape(-1)
                s_done = ep_done[idx].reshape(-1)
                s_action = ep_action[idx].reshape(-1,1)

                batch_queue.put((s_state,s_nx_state,s_reward,s_done,s_action))


    @torch.compile(mode="max-autotune")
    def compute_loss(self,s_nx_state,s_reward,s_done,pred_q):
        with torch.no_grad():
            nx_action = torch.argmax(self.q1(s_nx_state),dim=1).unsqueeze(-1)
            eval_ = self.target_net(s_nx_state).gather(1,nx_action).squeeze(1)
            target = s_reward + GAMMA * eval_ * (1 - s_done)
        return F.mse_loss(pred_q, target)
    

    def save(self,n): 
        data = { "q1 state":self.q1.state_dict(),
            "target net state":self.target_net.state_dict(),
            "optim state":self.optim.state_dict()
        }
        torch.save(data,f"{self.storage_path}/state_{n}.pth")


    def main(self):
        self.thread_1.start()
        self.thread_2.start()

        mlflow.set_experiment("pacman")
        with mlflow.start_run() as run:

            while self.batch_queue.qsize() < 10:
                time.sleep(0.1)

            for t in tqdm(range(MAX_STEPS//MAX_EP_STEPS),total=MAX_STEPS//MAX_EP_STEPS):
                s_state,s_nx_state,s_reward,s_done,s_action = self.batch_queue.get()
                
                s_state = s_state.to(DEVICE,non_blocking=True)
                s_nx_state = s_nx_state.to(DEVICE,non_blocking=True)
                s_reward = s_reward.to(DEVICE,non_blocking=True)
                s_done = s_done.to(DEVICE,non_blocking=True)
                s_action = s_action.to(DEVICE,non_blocking=True)

                with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16):
                    pred_q = self.q1(s_state).gather(1,s_action).squeeze(1)
                    loss = self.compute_loss(s_nx_state,s_reward,s_done,pred_q)

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                self.optim.step()

                if t > 0 and t % 2500 == 0:
                    self.target_net.load_state_dict(self.q1.state_dict())

                if t > 0 and t % 500 == 0:
                    self.save(t)
                    mlflow.log_metrics({
                        "average reward": self.reward_data.mean().item(),
                        "loss": loss.item(),
                    }, step=self.global_step) 

    
    def test(self):
        env = gym.make("ALE/MsPacman-v5",max_episode_steps=MAX_EP_STEPS,render_mode="human")
        env = GrayscaleObservation(env)
        env = ResizeObservation(env,R_SHAPE)
        env = FrameStackObservation(env,4)
        state = env.reset()[0]
        
        policy = q_function()
        checkpoint = torch.load("./state_6500.pth", map_location=torch.device("cpu"))
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
    
     






