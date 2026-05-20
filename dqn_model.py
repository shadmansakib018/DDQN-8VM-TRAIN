# dqn_model.py
import csv
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import defaultdict, deque
import os
import subprocess

class DQN(nn.Module):
    def __init__(self, input_dim=21, output_dim=8):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, path, state_dim=21, action_dim=8, gamma=0.99, lr=1e-3, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=1000000)
        self.lock = threading.Lock()
        self.train_lock = threading.Lock()
        self.epochs_per_train = 25   # Train for 25 epochs when triggered
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.98
        self.total_simulations = 0
        self.reward_summation = 0
        self.checkpoint_interval = 100  # Save every 100 simulations
        self.experience_log_path = "experience_log_150.csv"
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # self.reward_per_source = defaultdict(float)

        # Initialize the file and write header if it doesn't exist
        # if not os.path.exists(self.experience_log_path):
        #     with open(self.experience_log_path, mode="w", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["state", "action", "reward", "next_state"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)
        # self.policy_net = DQN(state_dim, action_dim).to(self.device)
        # self.target_net = DQN(state_dim, action_dim).to(self.device)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # self.loss_fn = nn.MSELoss()
        # self.update_count = 0

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)

        # Load checkpoint *before* creating optimizer
        self.load_checkpoint(path)

        # Make sure target_net matches the updated policy_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer should only include the parameters we actually want to train
        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.policy_net.parameters()), lr=lr)

        # ✅ 4. Create optimizer using only trainable params (e.g., final layer)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.policy_net.parameters()), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.update_count = 0

    def remember(self, state, action, reward, next_state, source_id):
        with self.lock:
            self.memory.append((state, action, reward, next_state))

        # if source_id is not None:
        self.reward_summation += reward
            # self.reward_per_source[source_id] += reward

        # with open(self.experience_log_path, mode="a", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow([ json.dumps(state),action,reward,json.dumps(next_state)])

    def train(self, source_id):
        with self.train_lock:
            if len(self.memory) < self.batch_size:
                return
            epoch_losses = 0
            for _ in range(self.epochs_per_train):
                        # Lock only while accessing shared memory
                with self.lock:
                    if len(self.memory) < self.batch_size:
                        return
                    batch = random.sample(self.memory, self.batch_size)

                states, actions, rewards, next_states = zip(*batch)

                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)

                q_values = self.policy_net(states).gather(1, actions)
                # next_q = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
                # DDQN: Use policy net for action selection, target net for evaluation
                with torch.no_grad():
                    next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                    next_q = self.target_net(next_states).gather(1, next_actions)

                target_q = rewards + self.gamma * next_q
                
                loss = self.loss_fn(q_values, target_q)
                epoch_losses = epoch_losses + loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.update_count += 1
            if self.update_count % 50 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            self.total_simulations += 1
            print("SIMULATION NUMBER: ",self.total_simulations)
            if(self.total_simulations >= 300 and self.total_simulations % self.checkpoint_interval==0):
                self.save_checkpoint()
        

            with open("loss_log.txt", "a") as f:
                f.write(f"{epoch_losses:.2f},{(self.reward_summation):.2f}\n")
            
            # self.reward_per_source[source_id] = 0
            self.reward_summation = 0



    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.policy_net(state).argmax().item()
    
    def save_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.total_simulations}.pth")
        torch.save(self.policy_net.state_dict(), checkpoint_path)
        print(f"[Checkpoint] Saved model to {checkpoint_path}")

        # Call validate.py and pass the checkpoint path
        try:
            subprocess.Popen(["python", "./validation/validate.py", checkpoint_path])
            print(f"[Validation] Started validation for {checkpoint_path}")
        except Exception as e:
            print(f"[Validation Error] Failed to start validation: {e}")


    # def load_checkpoint(self, path):
    #     self.policy_net.load_state_dict(torch.load(path))
    #     self.policy_net.eval()
    #     print(f"Loaded checkpoint from {path}")

    def load_checkpoint(self, path):
        # Load the pretrained weights from the 10-output model
        pretrained_dict = torch.load(path, map_location=self.device)

        # Get current model's state dict (which has 6 outputs)
        model_dict = self.policy_net.state_dict()

        # Filter out the keys of the final layer (last Linear layer)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

        # Update current model weights with matching pretrained weights
        model_dict.update(pretrained_dict)
        self.policy_net.load_state_dict(model_dict)

        # OPTIONAL: Freeze all layers except the final one
        for name, param in self.policy_net.named_parameters():
            if '4' not in name:  # Layer index 4 is the last Linear layer (32 → output_dim)
                param.requires_grad = False

        self.policy_net.eval()
        print(f"Loaded partial checkpoint (excluding output layer) from {path}")

    

    def load_latest_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir):
            return
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pth")]
        if not checkpoints:
            return
        latest = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        self.load_checkpoint(os.path.join(self.checkpoint_dir, latest))
        # self.load_checkpoint('C:/Users/ss4587s/Desktop/FlashProject/checkpoints/checkpoint_step_100.pth')


