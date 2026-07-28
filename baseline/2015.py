#Multi-threaded DDQN 

import gymnasium as gym 
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.wrappers.transform_observation import GrayscaleObservation,ResizeObservation
from gymnasium.wrappers import FrameStackObservation,ClipReward

import ale_py,torch,sys,random,mlflow,time
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from collections import deque
from tqdm import tqdm
from threading import Thread
from queue import Queue,Empty
from dataclasses import dataclass


@dataclass(frozen=True)
class hypers: # target max steps : 50 Million
    max_ep_steps = 500
    num_envs = 10
    r_shape = (100,100) 
    device = torch.device("cuda:0") 
    gamma = .99 
    lr = 25e-5
    batch_size = 32
    buffer_size = 10_000


class SkipFrame(gym.Wrapper):
    def __init__(self,env,skip):
        super().__init__(env)
        self.skip = skip

    def step(self,action):
        _rewards = 0
        for n in range(self.skip):
            observation,reward,done,truncation,info = self.env.step(action)
            _rewards += reward
            if done or truncation:
                break 
        return observation,_rewards,done,truncation,info

    def reset(self,**kwargs):
        observation,info = self.env.reset(**kwargs)
        return observation,info


def vec_env():
    def make():
        x = gym.make("ALE/MsPacman-v5",max_episode_steps=hypers().max_ep_steps)
        x = ResizeObservation(x,hypers().r_shape)
        x = GrayscaleObservation(x)
        x = SkipFrame(x,4)
        x = ClipReward(x,-1,1)
        x = FrameStackObservation(x,4)
        return x 
    return AsyncVectorEnv([make for _ in range(hypers().num_envs)])


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
        x = F.silu(self.c1(x))
        x = F.silu(self.c2(x))
        x = F.silu(self.c3(x)) 
        x = F.silu(self.l1(x.flatten(1)))
        return self.l2(x)


class ddqn:
    def __init__(self,storage_path="./"):
        self.hypers = hypers()
        self.storage_path = storage_path
        self.env = vec_env()
        self.channels = self.env.observation_space.shape[1]
        
        dummy_obs = (torch.randint(0,255,(self.hypers.num_envs,self.channels,*self.hypers.r_shape),dtype=torch.float))
        self.q1 = q_function()
        self.q1(dummy_obs)
        self.q1.to(self.hypers.device)

        self.q1_cpu = deepcopy(self.q1).cpu()
        self.target_net = deepcopy(self.q1).to(self.hypers.device)        
        self.q1.compile(mode="max-autotune") 
        
        self.optim = torch.optim.Adam(self.q1.parameters(),lr=self.hypers.lr,fused=True)
        self.reward_data = torch.zeros(self.hypers.num_envs,dtype=torch.float)
        self.global_step = 0

        self.episode_queue = Queue(maxsize=40) # holds full raw episodes
        self.batch_queue = Queue(maxsize=125)  # holds processed batches for GPU

 
    @torch.no_grad()
    def step_env(self,queue,run_id):
        t_state = torch.zeros(self.hypers.max_ep_steps,self.hypers.num_envs,self.channels,*self.hypers.r_shape,dtype=torch.uint8)
        t_nx_state = torch.zeros(self.hypers.max_ep_steps,self.hypers.num_envs,self.channels,*self.hypers.r_shape,dtype=torch.uint8)
        t_reward = torch.zeros(self.hypers.max_ep_steps,self.hypers.num_envs)
        t_done = torch.zeros(self.hypers.max_ep_steps,self.hypers.num_envs,dtype=torch.bool)
        t_action = torch.zeros(self.hypers.max_ep_steps,self.hypers.num_envs,dtype=torch.int64)

        state = torch.tensor(self.env.reset()[0],dtype=torch.float,device="cpu")
        n = 0

        while True:
            decay_fraction = min(self.global_step / int(1e5),1) # decay over 1 million / NUM ENVS
            epsilon = 1 - (1 - 0.1) * decay_fraction

            if random.random() < epsilon: action = self.env.action_space.sample()
            else: action = torch.argmax(self.q1_cpu(state),dim=1).tolist()

            nx_state, reward, done, trunc, _ = self.env.step(action)
            self.reward_data += reward

            t_state[n].copy_(state)
            t_nx_state[n].copy_(torch.as_tensor(nx_state))
            t_reward[n].copy_(torch.as_tensor(reward))
            t_done[n].copy_(torch.as_tensor(done))
            t_action[n].copy_(torch.as_tensor(action))

            state = torch.as_tensor(nx_state).float()

            n += 1
            self.global_step += 1

            if n == self.hypers.max_ep_steps:
                queue.put((t_state.clone(),t_nx_state.clone(),t_reward.clone(),t_done.clone(),t_action.clone()))
                
                mlflow.log_metrics({
                    "average reward":self.reward_data.mean().item(),
                    "epsiolon": epsilon },
                    step=self.global_step,run_id=run_id
                ) 
                n = 0
                self.reward_data = torch.zeros(self.hypers.num_envs,dtype=torch.float)
    

    def sample_processor(self,ep_queue,batch_queue):
        b_state = torch.zeros(self.hypers.buffer_size,self.hypers.num_envs,self.channels,*self.hypers.r_shape,dtype=torch.uint8)
        b_nx_state = torch.zeros(self.hypers.buffer_size,self.hypers.num_envs,self.channels,*self.hypers.r_shape,dtype=torch.uint8)
        b_reward = torch.zeros(self.hypers.buffer_size,self.hypers.num_envs)
        b_done = torch.zeros(self.hypers.buffer_size,self.hypers.num_envs,dtype=torch.bool)
        b_action = torch.zeros(self.hypers.buffer_size,self.hypers.num_envs,dtype=torch.int64)
        ptr = 0
        size = 0
        
        while True:

            while True:
                try:
                    ep_state,ep_nx_state,ep_reward,ep_done,ep_action = ep_queue.get_nowait()
            
                    indices = torch.arange(ptr, ptr + self.hypers.max_ep_steps) % self.hypers.buffer_size # circular buffer core indexing code 
                    b_state[indices] = ep_state.to(torch.uint8)
                    b_nx_state[indices] = ep_nx_state.to(torch.uint8)
                    b_reward[indices] = ep_reward
                    b_done[indices] = ep_done.to(torch.bool)
                    b_action[indices] = ep_action

                    ptr = (ptr + self.hypers.max_ep_steps) % self.hypers.buffer_size
                    size = min(size + self.hypers.max_ep_steps,self.hypers.buffer_size)
                except Empty:
                    break

            if size < self.hypers.max_ep_steps:
                time.sleep(0.001)
                continue
            
            if not batch_queue.full():
                for _ in range(self.hypers.max_ep_steps//4): # sampling every 4 step, 500 steps -> 125 samples
                    idx = torch.randint(0,size,(self.hypers.batch_size,))
                    env_idx = torch.randint(0,self.hypers.num_envs,(self.hypers.batch_size,))
             
                    s_state = b_state[idx,env_idx]       # torch.Size([32, 4, 100, 100])
                    s_nx_state = b_nx_state[idx,env_idx] # torch.Size([32, 4, 100, 100])
                    s_reward = b_reward[idx,env_idx]     # torch.Size([32]) 
                    s_done = b_done[idx,env_idx]         # torch.Size([32])
                    s_action = b_action[idx,env_idx].unsqueeze(-1) # torch.Size([32,1])
              
                    items = (s_state,s_nx_state,s_reward,s_done,s_action)
                    batch_queue.put(items)
            else:
                time.sleep(0.001)
                
                
    def save(self,n): 
        data = { 
            "q1 state":self.q1.state_dict(),
            "target net state":self.target_net.state_dict(),
            "optim state":self.optim.state_dict()
        }
        torch.save(data,f"{self.storage_path}/state_{n}.pth")


    @torch.compile(mode="max-autotune")
    def compute_loss(self,s_nx_state,s_reward,s_done,pred_q):
        with torch.no_grad():
            nx_action = torch.argmax(self.q1(s_nx_state),dim=1).unsqueeze(-1)
            eval_ = self.target_net(s_nx_state).gather(1,nx_action).squeeze(1)
            target = s_reward + self.hypers.gamma * eval_ * (1 - s_done)
        return F.mse_loss(pred_q,target)
    

    def main(self):
        mlflow.set_experiment("pacman")
        with mlflow.start_run() as run:

            run_id = run.info.run_id # to track average reward being tracking in the thread 1
            self.thread_1 = Thread(target=self.step_env,args=(self.episode_queue,run_id),daemon=True)
            self.thread_2 = Thread(target=self.sample_processor,args=(self.episode_queue,self.batch_queue),daemon=True)

            self.thread_1.start()
            while self.episode_queue.qsize() < 30: 
                time.sleep(0.01)

            self.thread_2.start()
  
            for t in tqdm(range((3_250_000) + 1),total=(3_250_000) + 1):
                s_state,s_nx_state,s_reward,s_done,s_action = self.batch_queue.get()

                s_state = s_state.to(self.hypers.device,torch.float32)
                s_nx_state = s_nx_state.to(self.hypers.device,torch.float32)
                s_reward = s_reward.to(self.hypers.device)
                s_done = s_done.to(self.hypers.device,torch.float32)
                s_action = s_action.to(self.hypers.device)

                pred_q = self.q1(s_state).gather(1,s_action).squeeze(1)
                loss = self.compute_loss(s_nx_state,s_reward,s_done,pred_q)

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                self.optim.step()

                if t > 0 and t % 2_500 == 0:
                    self.target_net.load_state_dict(self.q1.state_dict()) 
             
                if t > 0 and t % 500 == 0: 
                    mlflow.log_metric("loss",loss.item(),step=t) 
                    self.q1_cpu.load_state_dict(self.q1.state_dict()) # update cpu version weights

                if t > 0 and t % 50_000 == 0:
                    self.save(t//50_000)
            
            self.save(t)

    
    def test(self):
        env = gym.make("ALE/MsPacman-v5",render_mode="human")
        env = GrayscaleObservation(env)
        env = ResizeObservation(env,self.hypers.r_shape)
        env = SkipFrame(env,4)
        env = FrameStackObservation(env,4)
        state = env.reset()[0]
        
        policy = q_function()
        checkpoint = torch.load("./2015_model.pth",map_location=torch.device("cpu"))
        compiled_state_dict = {k.replace("_orig_mod.",""): v for k, v in checkpoint["q1 state"].items()}
        policy.load_state_dict(compiled_state_dict)
        rewards = 0 
        while True:
            state = torch.tensor(state,dtype=torch.float).unsqueeze(0) 
            nx_s,reward,done,trunc,info = env.step(torch.argmax(policy(state)).item())
            state = nx_s
            rewards += reward
            env.render()
            if done or trunc:
                print(rewards)
                break


if __name__ == "__main__": 
    import warnings,logging
    warnings.filterwarnings("ignore") ; logging.disable(logging.CRITICAL)
    ddqn("./").test()
