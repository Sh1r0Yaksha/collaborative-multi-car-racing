import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from environment import MultiCarRacing
import random
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


# Experience replay memory tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.LongTensor([exp.action for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.FloatTensor([exp.done for exp in experiences])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class MADQN:
    def __init__(self, n_agents, state_dim, action_dim, writer):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.writer = writer
        
        # Create networks and replay buffers for each agent
        self.q_networks = {}
        self.target_networks = {}
        self.optimizers = {}
        self.replay_buffers = {}
        
        for agent_id in range(n_agents):
            self.q_networks[agent_id] = DQNetwork(state_dim, action_dim)
            self.target_networks[agent_id] = DQNetwork(state_dim, action_dim)
            self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())
            self.optimizers[agent_id] = optim.Adam(self.q_networks[agent_id].parameters())
            self.replay_buffers[agent_id] = ReplayBuffer(capacity=10000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.gamma = 0.95
        self.batch_size = 64
        self.target_update_frequency = 10
        self.update_counter = 0
        self.training_step = 0
    
    def select_action(self, states):
        actions = {}
        for agent_id in range(self.n_agents):
            if random.random() < self.epsilon:
                actions[agent_id] = random.randint(0, self.action_dim - 1)
            else:
                state = torch.FloatTensor(states[agent_id]).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.q_networks[agent_id](state)
                    actions[agent_id] = q_values.argmax().item()
        return actions
    
    def update(self, agent_id):
        if len(self.replay_buffers[agent_id]) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffers[agent_id].sample(self.batch_size)
        
        # Compute current Q values
        current_q_values = self.q_networks[agent_id](states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_networks[agent_id](next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Log loss to TensorBoard
        self.writer.add_scalar(f'Loss/agent_{agent_id}', loss.item(), self.training_step)
        self.writer.add_scalar(f'Q_values/agent_{agent_id}', current_q_values.mean().item(), self.training_step)
        
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_step += 1

def train_madqn():
    # Create unique run name with timestamp
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'MADQN_{current_time}')
    writer = SummaryWriter(log_dir)
    
    env = MultiCarRacing(n_cars=2, grid_size=30, track_width=5, num_checkpoints=12, render_mode=None)
    n_agents = 2
    state_dim = 30 * 30  #observation space is the grid size
    action_dim = 5  # Number of possible actions - up, down, left, right and stay
    
    # Initialize MADQN with TensorBoard writer
    madqn = MADQN(n_agents, state_dim, action_dim, writer)
    
    n_episodes = 1000
    max_steps = 500
    
    # Lists to store metrics for plotting
    all_episode_rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_rewards = {i: 0 for i in range(n_agents)}
        episode_steps = 0
        
        for step in range(max_steps):
            episode_steps += 1
            states = {i: obs[i].flatten() for i in range(n_agents)}
            actions = madqn.select_action(states)
            next_obs, rewards, dones, info = env.step(actions)
            
            # Store experiences and update
            for agent_id in range(n_agents):
                madqn.replay_buffers[agent_id].push(
                    states[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_obs[agent_id].flatten(),
                    dones[agent_id]
                )
                episode_rewards[agent_id] = rewards[agent_id]
        
            
            for agent_id in range(n_agents):
                madqn.update(agent_id)
            
            obs = next_obs
            env.render()
            
            if any(dones.values()):
                break
        
        # Log episode metrics to TensorBoard
        for agent_id in range(n_agents):
            writer.add_scalar(f'Rewards/agent_{agent_id}', episode_rewards[agent_id], episode)
            writer.add_scalar(f'Steps/agent_{agent_id}', episode_steps, episode)
        
        writer.add_scalar('Training/epsilon', madqn.epsilon, episode)
        writer.add_scalar('Training/episode_length', episode_steps, episode)
        
        # Calculate and log average reward across all agents
        avg_reward = sum(episode_rewards.values()) / n_agents
        writer.add_scalar('Rewards/average', avg_reward, episode)
        
        # Log checkpoints reached and other custom metrics if available in info
        if info:
            for agent_id in range(n_agents):
                if 'checkpoints_reached' in info:
                    writer.add_scalar(f'Checkpoints/agent_{agent_id}', 
                                    info['checkpoints_reached'].get(agent_id, 0), 
                                    episode)
        
        # Print episode statistics
        print(f"Episode {episode + 1}")
        for agent_id in range(n_agents):
            print(f"Agent {agent_id} total reward: {episode_rewards[agent_id]}")
        print(f"Epsilon: {madqn.epsilon}")
        print("--------------------")
    
    writer.close()

if __name__ == "__main__":
    train_madqn()
    
    
