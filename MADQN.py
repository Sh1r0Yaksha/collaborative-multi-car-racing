import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import copy
from environment import MultiCarRacing
# Experience replay memory tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.tensor(np.array([exp.state for exp in experiences]), dtype=torch.float32)
        actions = torch.tensor(np.array([exp.action for exp in experiences]), dtype=torch.long)
        rewards = torch.tensor(np.array([exp.reward for exp in experiences]), dtype=torch.float32)
        next_states = torch.tensor(np.array([exp.next_state for exp in experiences]), dtype=torch.float32)
        dones = torch.tensor(np.array([exp.done for exp in experiences]), dtype=torch.float32)
        # print(states, "where ")
        # print(actions,"accc ")
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNetwork, self).__init__()

        self.input_channels = input_shape[2]  # Number of channels (RGB ie 3)
        self.input_height = input_shape[0]    # Grid height
        self.input_width = input_shape[1]     # Grid width

        # CNN layers
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=2)  # Adjusted kernel size and stride
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.input_width, 5, 2), 3, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.input_height, 5, 2), 3, 2), 3, 1)

        if convw <= 0 or convh <= 0:
            raise ValueError("Convolutional layers reduce input dimensions below valid size. Adjust kernel/stride.")

        linear_input_size = convw * convh * 64

        # FC layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # Print input shape for debugging
        print(f"Input shape: {x.shape}")
        x = x.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
        x = torch.relu(self.conv1(x))
        print(f"After conv1: {x.shape}")
        x = torch.relu(self.conv2(x))
        print(f"After conv2: {x.shape}")
        x = torch.relu(self.conv3(x))
        print(f"After conv3: {x.shape}")

        # Flatten the output of the convolution layers
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MADQNAgent:
    def __init__(self, state_shape, n_actions, agent_id):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.agent_id = agent_id
        
        # Networks
        self.policy_net = DQNetwork(state_shape, n_actions).to(self.device)
        self.target_net = DQNetwork(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayBuffer(100000)
        
        # Hyperparameters
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Add gradient clipping
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

class MADQNTrainer:
    def __init__(self, env, n_agents):
        self.env = env
        self.n_agents = n_agents
        
        # Get state and action space information from environment
        state_shape = env.observation_space("car_0").shape
        n_actions = env.action_space("car_0").n
        
        # Create agents
        self.agents = {
            f"car_{i}": MADQNAgent(state_shape, n_actions, f"car_{i}")
            for i in range(n_agents)
        }
        
    def train(self, n_episodes):
        best_reward = float('-inf')
        episode_rewards = []

        for episode in range(n_episodes):
            episode_reward = 0
            step = 0

            # Reset environment
            self.env.reset()

            # Get initial state for first agent
            state = self.env.observe(self.env.agent_selection)

            while self.env.agents:  # Continue while there are active agents
                step += 1
                current_agent = self.env.agent_selection

                # Check if the agent is alive (not dead) before selecting and performing action
                if not self.env.terminations[current_agent] and not self.env.truncations[current_agent]:
                    # Select and perform action
                    action = self.agents[current_agent].select_action(state)
                else:
                    # Set action to None if agent is dead
                    action = None

                last_state = state
                self.env.step(action)  # Step the environment with the action
                next_state = self.env.observe(current_agent)
                reward = self.env.rewards[current_agent]
                done = self.env.terminations[current_agent] or self.env.truncations[current_agent]

                # Store transition in memory
                self.agents[current_agent].memory.push(
                    last_state, action, reward, next_state, done
                )

                # Learn
                if len(self.agents[current_agent].memory) >= self.agents[current_agent].batch_size:
                    loss = self.agents[current_agent].learn()

                # Update target
                if step % self.agents[current_agent].target_update == 0:
                    self.agents[current_agent].target_net.load_state_dict(
                        self.agents[current_agent].policy_net.state_dict()
                    )

                # Update metrics
                episode_reward += reward
                state = next_state  # Update state for next iteration

                # Check if all agents are done
                if all(self.env.terminations.values()) or all(self.env.truncations.values()):
                    break

            # Track episode rewards
            episode_rewards.append(episode_reward)

            # Print episode summary
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode + 1}")
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"Average Reward (last 100): {avg_reward:.2f}")
            print(f"Epsilon: {self.agents['car_0'].epsilon:.2f}")

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_models("best_models")

        return episode_rewards

    
    def save_models(self, path):
        """Save all agents' models"""
        import os
        os.makedirs(path, exist_ok=True)
        for agent_id, agent in self.agents.items():
            torch.save(agent.policy_net.state_dict(), f"{path}/{agent_id}_policy.pth")
            torch.save(agent.targenvet_net.state_dict(), f"{path}/{agent_id}_target.pth")
    
    def load_models(self, path):
        """Load all agents' models"""
        for agent_id, agent in self.agents.items():
            agent.policy_net.load_state_dict(torch.load(f"{path}/{agent_id}_policy.pth"))
            agent.target_net.load_state_dict(torch.load(f"{path}/{agent_id}_target.pth"))


def train_agents(render_mode=None, n_agents=4, n_episodes=1000):
    environment = MultiCarRacing(n_cars=n_agents, render_mode=render_mode)
    trainer = MADQNTrainer(environment, n_agents)
    rewards = trainer.train(n_episodes)
    trainer.save_models("final_models")
    return trainer, rewards

if __name__ == "__main__":
    trainer, rewards = train_agents(render_mode="human", n_agents=1, n_episodes=1000)

