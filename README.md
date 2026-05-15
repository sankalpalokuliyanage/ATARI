# DQN Atari Agent: Playing Pong with Deep Reinforcement Learning

This repository contains a PyTorch implementation of the landmark 2013 DeepMind paper: **"Playing Atari with Deep Reinforcement Learning"** by Mnih et al. 

The agent utilizes a **Deep Q-Network (DQN)** to learn how to play the Atari 2600 game *Pong* directly from raw screen pixels, eventually achieving and surpassing human-level performance.

---

## 🚀 Key Features

* **End-to-End Learning:** The agent has no prior knowledge of the game rules; it learns solely from the raw pixel input and the rewards provided by the environment.
* **Experience Replay:** Stores 1,000,000 recent transitions in a replay buffer. Training on random minibatches from this buffer breaks the correlation between consecutive samples and stabilizes training.
* **Target Network:** Implements a separate target network to provide stable Q-value targets during the optimization process.
* **Frame Stacking:** To understand movement and velocity, the agent receives a stack of the 4 most recent preprocessed frames.
* **Reward Clipping:** All positive rewards are clipped to +1 and negative rewards to -1 to ensure consistent gradient scales across different games.

---

## 🛠️ Hardware Specifications

This project was trained and optimized on a high-performance workstation:
* **GPU:** NVIDIA RTX A4500 (16GB VRAM) - Handles high-speed parallel CNN computations.
* **RAM:** 256GB DDR4 - Efficiently manages the massive 1M experience replay memory.
* **OS:** Windows 11 / Linux compatible.

---

## 📁 Project Structure

* `train.py`: Main script for training the agent using the DQN algorithm.
* `test.py`: Script to load the trained model weights and watch the agent play in real-time.
* `models/`: Folder containing saved model checkpoints (`.pth` files).
* `README.md`: Documentation of the project.

---

## 🧠 Technical Overview & Mathematics

The agent approximates the optimal **Action-Value Function** using a Convolutional Neural Network (CNN).

### 1. The Neural Network (CNN)
The architecture consists of:
* 2 Convolutional layers (capturing visual patterns).
* 1 Fully connected layer (256 units).
* Linear output layer (one output per possible game action).



### 2. The Bellman Equation
The core of the learning process is the **Bellman Equation**, used to estimate the value of a state-action pair:
$$Q(s, a) \approx r + \gamma \max_{a'} Q(s', a')$$
*where $r$ is the reward, and $\gamma$ (0.99) is the discount factor.*



### 3. Optimization
Training minimizes the **Mean Squared Error (MSE) Loss** between the predicted Q-value and the target Q-value:
$$L(\theta) = \mathbb{E} [(Target\_Q - Predicted\_Q)^2]$$

---

## 📋 Requirements & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sankalpalokuliyanage/ATARI.git
    cd atari-main
    ```

2.  **Install dependencies:**
    ```bash
    pip install gymnasium[atari] gymnasium[accept-rom-license] ale-py torch torchvision opencv-python numpy
    ```

3.  **Setup Atari ROMs:**
    ```bash
    AutoROM --accept-license
    ```

---

## 🎮 How to Run

### Training
To start training the agent from scratch or resume from a checkpoint:
```bash
python train.py
```

###Testing / Evaluation
To watch your trained agent play after training (requires a .pth file in the models/ folder):
```bash
python test.py
```

##📊 Training Results
*Exploration: Epsilon starts at 1.0 (pure exploration) and decays to 0.1 over 1 million frames.

*Performance: The agent typically starts outscoring the built-in Atari AI after approximately 700-1000 episodes.

*Max Score: In Pong, the agent can achieve a maximum score of +21.

##Developed by
L.C. Sankalpa Lokuliyanage

##📜 References
Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2013). Playing Atari with Deep Reinforcement Learning.

##📝 Acknowledgments
Kyungpook National University (KNU) for providing the computational resources.


