import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import cv2
import os
from collections import deque

# Registering ALE with Gymnasium for Atari access [cite: 106]
gym.register_envs(ale_py)

# --- Hyperparameters as per Section 5 [cite: 181-182] ---
GAMMA = 0.99
LEARNING_RATE = 0.00025
REPLAY_SIZE = 1000000   # Replay memory capacity N [cite: 127]
BATCH_SIZE = 32         # Stochastic minibatch size [cite: 181]
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000 # Epsilon annealed over 1M frames [cite: 181]
TARGET_UPDATE = 10000   # Frequency for Target Network update
SAVE_INTERVAL = 50      # Save every 50 episodes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Preprocessing [cite: 155-156] ---
def preprocess_frame(frame):
    """ Converts RGB to 84x84 grayscale [cite: 155-156]. """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(resized, dtype=np.uint8)

# --- DQN Architecture [cite: 167-171] ---
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        # Layers defined in paper [cite: 168-171]
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256) # 256 rectifier units
        self.output = nn.Linear(256, action_size)

    def forward(self, x):
        x = x.float() / 255.0 # Normalizing pixels [cite: 114]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.output(x)

# --- Training Agent ---
class AtariAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_SIZE)
        self.epsilon = EPSILON_START
        self.policy_net = DQN(action_size).to(DEVICE)
        self.target_net = DQN(action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.steps = 0

    def select_action(self, state):
        """ Epsilon-greedy action selection [cite: 130-131]. """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return self.policy_net(state_t).argmax().item()

    def optimize(self):
        """ Minibatch update from Experience Replay D [cite: 135-139]. """
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            expected_q = rewards + (GAMMA * max_next_q * (1 - dones)) # Bellman Eq [cite: 61, 136]

        loss = nn.MSELoss()(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_END:
            self.epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Successfully loaded existing model: {path}")
            return True
        return False

# --- Main Logic ---
def main(game_id="ALE/Pong-v5", load_latest=True):
    env = gym.make(game_id, render_mode="rgb_array")
    agent = AtariAgent(env.action_space.n)
    model_path = f"models/dqn_{game_id.replace('/', '_')}.pth"
    
    if not os.path.exists("models"):
        os.makedirs("models")

    # Try to load existing model before starting
    if load_latest:
        agent.load_model(model_path)

    print(f"Status: Using {DEVICE} for training on {game_id}")

    for episode in range(1, 20001):
        obs, _ = env.reset()
        state = preprocess_frame(obs)
        state_stack = deque([state] * 4, maxlen=4) # Frame stacking [cite: 158]
        total_reward = 0
        
        while True:
            current_state = np.stack(state_stack)
            action = agent.select_action(current_state)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            clipped_reward = np.sign(reward) # Reward clipping [cite: 178]
            next_f = preprocess_frame(next_obs)
            next_stack = list(state_stack)[1:] + [next_f]
            next_state = np.stack(next_stack)

            agent.memory.append((current_state, action, clipped_reward, next_state, done))
            state_stack.append(next_f)
            total_reward += reward
            
            agent.optimize()
            agent.steps += 1
            
            if agent.steps % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            if done:
                print(f"Episode: {episode} | Score: {total_reward} | Epsilon: {agent.epsilon:.2f}")
                break
        
        if episode % SAVE_INTERVAL == 0:
            agent.save_model(model_path)

    env.close()

if __name__ == "__main__":
    main()