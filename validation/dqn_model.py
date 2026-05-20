import torch
import torch.nn as nn
import torch.optim as optim

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
        self.path = path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.update_count = 0

        self.load_checkpoint(path)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.policy_net(state).argmax().item()

    def load_checkpoint(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
        print(f"Loaded checkpoint from {path}")


