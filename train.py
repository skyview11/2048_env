import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import random




EPISODES = 50000
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = BATCH_SIZE
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995


class Memory:
    def __init__(self, mem_size, data_dim):
        self.data_dim = data_dim
        self.mem_size = mem_size
        self.memory = [torch.zeros(mem_size, dim) for dim in data_dim]
        self.n = 0
    def append(self, data):
        for i, d in enumerate(data):
            self.memory[i][self.n] = d
        self.n  = (self.n + 1)%self.mem_size
    
    def sample(self, k):
        indices = random.sample(range(k), k)
        return [self.memory[i][indices] for i in range(len(self.data_dim))]
    def __len__(self):
        return self.n+1
            

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
    def forward(self, x):
        return self.net(x)
    
class DQNAgent:
    def __init__(self, state_dim, action_dim, model_path=None):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.memory = Memory(MEMORY_SIZE, (self.state_dim, 1, 1, self.state_dim, 1))
        self.epsilon = EPS_START
        
        self.model = DQN(state_dim, action_dim).to("cuda")
        self.target_model = DQN(state_dim, action_dim).to("cuda")
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        
        if model_path is not None:
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            self.target_model.load_state_dict(state_dict)
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            q_values = self.model(state.to("cuda"))
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # batch = random.sample(self.memory, BATCH_SIZE)
        # import pdb;pdb.set_trace()
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = states.to("cuda")
        actions = actions.long().to("cuda")
        rewards = rewards.squeeze().to("cuda")
        next_states = next_states.to("cuda")
        dones = dones.squeeze().to("cuda")
        
        # Q(s, a)
        curr_Q = self.model(states).gather(1, actions).squeeze()
        
        # target Q -> GT
        next_Q = self.target_model(next_states).max(1)[0].detach()
        target_Q = rewards + GAMMA * next_Q * (1 - dones)
        
        # import pdb;pdb.set_trace()
        loss = self.loss_fn(curr_Q, target_Q)
        loss_avg = torch.mean(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_avg
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        

USE_WANDB = True

exp_name = "run2_continue"


def encode_state(state):
    encoded_boardstate = state + (state==-1)*2
    encoded_boardstate = torch.log2(encoded_boardstate)
    encoded_boardstate = encoded_boardstate - max(encoded_boardstate.min(), 1) + 1
    # encoded_boardstate = encoded_boardstate / encoded_boardstate.max()
    return encoded_boardstate

if __name__ == "__main__":
    from env import Env2048
    import tqdm
    
    env = Env2048(1, (4,4))
    
    state_dim = 16
    action_dim = 4
    
    agent = DQNAgent(state_dim, action_dim, "exp/run2/epoch_latest.pth")
    
    if USE_WANDB:
        import wandb
        wandb.init(project="2048_rl", name=exp_name)
        
    import os
    os.makedirs(f"exp/{exp_name}", exist_ok=True)

       
    
    # for episode in (range(EPISODES)):
    for episode in tqdm.tqdm(range(EPISODES)):
        if episode != 0:
            env.reset()
        state = encode_state(env.boardstate)
        total_reward = 0
        moved = 0
        not_moved = 0
        n_step = 0
        total_loss = 0
        for t in range(32*32 * (episode//1000 + 1)):
            action = agent.act(state)
            
            info = env.step([action])
            next_state, reward, done = info["state"][0], float(info["score"][0]), float(info["done"][0])
            next_state = encode_state(torch.tensor(next_state))
            
            ##################################################
            ## reward tuning
            
            ## 기본 공식
            reward = reward * 0.01
            # if done: ## 패배시 패널티
            #     # reward -= 1000 * (episode//100 + 1)
            #     reward -= 1000
            ## 못움직이는 행동하면 패널티
            if torch.all(next_state==state):
                # import pdb;pdb.set_trace()
                reward -= 0.001
                not_moved += 1
            else:
                moved += 1
                
            ####################################################
                
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                total_loss += loss
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        n_step = t
        agent.update_target()
        agent.epsilon = max(EPS_END, agent.epsilon * EPS_DECAY)

        info = {"Episode": episode + 1, 
                "Total_Reward": (total_reward), 
                "Epsilon": (agent.epsilon), 
                "steps": n_step, 
                "moved_ratio": (moved/(moved+not_moved)),
                "loss_avg": (float(total_loss)/n_step)}
        
        if episode % 100 == 0 and episode != 0:
            torch.save(agent.model.state_dict(), f"exp/{exp_name}/epoch_{episode}.pth")
            torch.save(agent.model.state_dict(), f"exp/{exp_name}/epoch_latest.pth")
        if USE_WANDB:
            wandb.log(info)
        else:
            print(info)
        
        # print(f"Episode {episode + 1}: Total Reward = {total_reward:.1f}, Epsilon = {agent.epsilon:.2f}, steps = {n_step:.2f}")
    # import pdb;pdb.set_trace()
    if USE_WANDB:
        wandb.save("model.pt")
        wandb.finish()