import matplotlib.pyplot as plt
import pandas as pd
import datetime

# # Load the loss log
# df = pd.read_csv("loss_log.txt", header=None, names=["TotalLoss", "RewardSummation"])

# # Load the long term rewards (average response times)
# with open("long_term_rewards.txt", "r") as f:
#     long_term_rewards = [float(line.strip()) for line in f]

# # Sanity check: ensure same number of entries
# assert len(df) == len(long_term_rewards), "Mismatch between loss_log.txt and long_term_rewards.txt!"

# # Subtract long_term_reward from reward_summation
# adjusted_rewards = df["RewardSummation"] - long_term_rewards

# # Compute running averages (Exponential Moving Average)
# alpha = 0.1  # Smoothing factor

# running_loss = [df["TotalLoss"][0]]
# running_reward = [adjusted_rewards[0]]

# for i in range(1, len(df)):
#     new_loss = alpha * df["TotalLoss"][i] + (1 - alpha) * running_loss[-1]
#     new_reward = alpha * adjusted_rewards[i] + (1 - alpha) * running_reward[-1]
#     running_loss.append(new_loss)
#     running_reward.append(new_reward)

# # Timestamp for saving plots
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# # Plot 1: Total Loss and Running Loss
# plt.figure(figsize=(10, 6))
# # plt.plot(range(1, len(df)+1), df["TotalLoss"], label="Total Loss", alpha=0.5)
# plt.plot(range(1, len(df)+1), running_loss, label="Running Loss (EMA)", linewidth=2)
# plt.xlabel('Simulation Run')
# plt.ylabel('Loss')
# plt.title('Loss and Running Loss Across Simulation Runs')
# plt.legend()
# plt.grid()
# filename = f"images/loss_plot_{timestamp}.png"
# plt.savefig(filename, dpi=300, bbox_inches='tight')

# # Plot 2: Adjusted Reward and Running Reward
# plt.figure(figsize=(10, 6))
# # plt.plot(range(1, len(df)+1), adjusted_rewards, label="Adjusted Reward", alpha=0.5, color='green')
# plt.plot(range(1, len(df)+1), running_reward, label="Running Reward (EMA)", color='orange', linewidth=2)
# plt.xlabel('Simulation Run')
# plt.ylabel('Reward')
# plt.title('Reward and Running Reward Across Simulation Runs')
# plt.legend()
# plt.grid()
# filename = f"images/reward_plot_{timestamp}.png"
# plt.savefig(filename, dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()

# print("Plots saved successfully!")


import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Load the loss log
df = pd.read_csv("loss_log.txt", header=None, names=["TotalLoss", "RewardSummation"])

# Load the long term rewards (average response times)
# with open("long_term_rewards.txt", "r") as f:
#     long_term_rewards = [float(line.strip()) for line in f]

# Sanity check: ensure same number of entries
# assert len(df) == len(long_term_rewards), "Mismatch between loss_log.txt and long_term_rewards.txt!"

# Subtract long_term_reward from reward_summation
# adjusted_rewards = df["RewardSummation"] - long_term_rewards
adjusted_rewards = df["RewardSummation"]

# Compute running averages (Exponential Moving Average)
alpha = 0.1  # Smoothing factor

running_loss = [df["TotalLoss"][0]]
running_reward = [adjusted_rewards[0]]


for i in range(1, len(df)):
    new_loss = alpha * df["TotalLoss"][i] + (1 - alpha) * running_loss[-1]
    new_reward = alpha * adjusted_rewards[i] + (1 - alpha) * running_reward[-1]
    running_loss.append(new_loss)
    running_reward.append(new_reward)

# Timestamp for saving plots
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Plot 1: Total Loss and Running Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(df)+1), df["TotalLoss"], label="Total Loss", alpha=0.5)
plt.plot(range(1, len(df)+1), running_loss, label="Running Loss", linewidth=2)
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.title('Loss and Running Loss')
plt.legend()
plt.grid()
filename = f"images/loss_plot_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')

# Plot 2: Adjusted Reward and Running Reward
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(df)+1), adjusted_rewards, label="Reward", alpha=0.5, color='green')
plt.plot(range(1, len(df)+1), running_reward, label="Running Reward", color='orange', linewidth=2)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward and Running Reward')
plt.legend()
plt.grid()
filename = f"images/reward_plot_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Plots saved successfully!")
