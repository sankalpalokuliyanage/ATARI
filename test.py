import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque
import time

# Registering ALE
gym.register_envs(ale_py)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.output = nn.Linear(256, action_size)

    def forward(self, x):
        x = x.float() / 255.0
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.output(x)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(resized, dtype=np.uint8)

def test(game_id="ALE/Pong-v5", model_path="models/dqn_ALE_Pong-v5.pth"):
   
    env = gym.make(game_id, render_mode="human")
    action_size = env.action_space.n
    
    # Load Model
    model = DQN(action_size).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Set to evaluation mode
    
    print(f"Testing Model: {model_path} on {game_id}")

    for episode in range(5): 
        obs, _ = env.reset()
        state = preprocess_frame(obs)
        state_stack = deque([state] * 4, maxlen=4)
        total_reward = 0
        done = False
        
        while not done:
            current_state = np.stack(state_stack)
            state_t = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            
            with torch.no_grad():
                action = model(state_t).argmax().item()
            
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            next_f = preprocess_frame(next_obs)
            state_stack.append(next_f)
            total_reward += reward
            
            
            time.sleep(0.01) 

        print(f"Test Episode: {episode + 1} | Final Score: {total_reward}")

    env.close()

if __name__ == "__main__":
    test()