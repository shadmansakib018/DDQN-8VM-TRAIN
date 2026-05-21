import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dim=21, output_dim=8):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)


class PPOAgent:
    def __init__(self, path, state_dim=21, action_dim=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.ac_net = ActorCritic(state_dim, action_dim).to(self.device)
        self.load_checkpoint(path)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.ac_net(state)         # critic output ignored in validation
            action = Categorical(logits=logits).probs.argmax().item()  # greedy, no sampling
        return action

    def load_checkpoint(self, path):
        self.ac_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.ac_net.eval()
        print(f"Loaded checkpoint from {path}")