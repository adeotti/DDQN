"""
Double Q-learning (Hasselt, 2010)
https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html

environment : cliff walking
action space : 4
observation space : 48
"""

import gymnasium as gym 
import torch,sys,random
import matplotlib.pyplot as plt
from tqdm import tqdm


class ddqn:
    
    n_ep = 250     # number of episodes
    horizon = 200  # steps per episodes
    u_prob = 0.5   # update probability
    alpha = 0.3    # step size
    gamma = 0.98
    power = 0.8

    def __init__(self):
        self.env = gym.make("CliffWalking-v1",max_episode_steps=self.horizon)
        self.o_s = self.env.observation_space.n  # obs space shape
        self.a_s = self.env.action_space.n       # action space shape

        self.q_a = torch.zeros((self.o_s,self.a_s),dtype=torch.float)
        self.q_b = torch.zeros((self.o_s,self.a_s),dtype=torch.float)
        self.visit_count = torch.zeros((self.o_s,),dtype=torch.float)

        self.n_a = torch.zeros((self.o_s,self.a_s),dtype=torch.float) # update count of q_a 
        self.n_b = torch.zeros((self.o_s,self.a_s),dtype=torch.float) # update count of q_b
        
        self.get_epsilon = lambda x : 1/torch.sqrt(x)
        self.get_step = lambda x : 1/torch.pow(x,self.power)
        self.r = 0
        self.r_data = []
        self.loss_data = []
  
    def main(self):
        for n in tqdm(range(self.n_ep),total=self.n_ep):

            state = self.env.reset()[0]
            for i in range(self.horizon):
                self.visit_count[state] += 1
                epsilon = self.get_epsilon(self.visit_count[state])
            
                if random.random() < epsilon.item():
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(self.q_a[state] + self.q_b[state]).tolist()
                
                nx_state,reward,done,trunc,_ = self.env.step(action) 
                 
                if random.random() > self.u_prob: 
                    a = torch.argmax(self.q_a[nx_state]) # a*
                    a_eval = self.q_b[nx_state,a] 

                    pred = self.q_a[state,action]
                    target = reward + (self.gamma * a_eval * (1-trunc))
                    loss = target - pred

                    self.n_a[state,action] +=1
                    step_a = self.get_step(self.n_a[state,action])
                    self.q_a[state,action] += (step_a * loss)
                    assert torch.all(torch.isfinite(self.q_a)), f"{self.q_a[state]}"
                else:
                    b = torch.argmax(self.q_b[nx_state]) # b*
                    b_eval = self.q_a[nx_state,b]

                    pred = self.q_b[state,action]
                    target = reward + (self.gamma * b_eval * (1-trunc))
                    loss = target - pred
                    
                    self.n_b[state,action] += 1
                    step_b = self.get_step(self.n_b[state,action])
                    self.q_b[state,action] += (step_b * loss)
                    assert torch.all(torch.isfinite(self.q_b)), f"{self.q_b[state]}"

                state = nx_state
                self.r += reward
   
                if trunc:
                    break
            
            self.loss_data.append(loss.item())
            self.r_data.append(self.r) # tracking rewards per episodes
            self.r = 0
                                  
        return self.q_a,self.q_b,[self.loss_data,self.r_data]


    def test(self):
        q_a,q_b,logs = self.main()
        self.env = gym.make("CliffWalking-v1",render_mode="human")
        state = self.env.reset()[0]
        
        for n in range(self.horizon):
            action = torch.argmax(q_a[state] + q_b[state]).tolist()

            nx_state,_,done,trunc,_ = self.env.step(action)
            state = nx_state
            self.env.render()

            if done or trunc:
                break
            
        
if __name__ == "__main__":
    ddqn().test() 
