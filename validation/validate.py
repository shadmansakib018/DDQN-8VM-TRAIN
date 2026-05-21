from flask_server import create_flask_app
from dqn_model import DQNAgent
import threading
import subprocess
import time
import sys
import os

from ppo_model import PPOAgent

# --- Configuration ---
checkpoint_path = sys.argv[1]  # path passed from main.py
BASE_PORT = 3000
# jar_path = "DDQN_VALIDATE.jar"
jar_path = os.path.join(os.path.dirname(__file__), "PPO_VALIDATE_8VM.jar")
num_epochs = 50
batch_size = 50
lb = 4

avg_file = os.path.join(os.path.dirname(__file__), "ValidationART.txt")
best_file =  os.path.join(os.path.dirname(__file__), "best_model.txt")

# --- Load Model ---
# agent = DQNAgent(checkpoint_path)
agent = PPOAgent(checkpoint_path)

# --- Start Flask Server ---
app = create_flask_app(agent, BASE_PORT)
thread = threading.Thread(target=app.run, kwargs={"port": BASE_PORT})
thread.daemon = True
thread.start()

print(f"🧠 Flask server running on port {BASE_PORT}")
time.sleep(2)

# --- Run Validation JAR ---
proc = subprocess.Popen([
    "java", "-jar", jar_path,
    str(BASE_PORT), str(batch_size), str(lb), str(num_epochs), checkpoint_path
])
proc.wait()

print("✅ Validation Complete")

# --- Read Result --- for DDQN
# try:
#     with open(avg_file, "r") as f:
#         avg_response_time = float(f.read().strip())
# except Exception as e:
#     print(f"⚠️ Error reading {avg_file}: {e}")
#     sys.exit(1)

# --- Read Result --- for PPO validate
try:
    with open(avg_file, "r") as f:
        lines = f.read().splitlines()
        lines = [l for l in lines if l.strip()]  # ignore empty lines
        avg_response_time = float(lines[-1])      # always grab the latest
except Exception as e:
    print(f"⚠️ Error reading {avg_file}: {e}")
    sys.exit(1)

# --- Load Previous Best ---
best_score = float('inf')
if os.path.exists(best_file):
    try:
        with open(best_file, "r") as f:
            line = f.read().strip()
            if line:
                _, best_score_str = line.split(",")
                best_score = float(best_score_str)
    except Exception as e:
        print(f"⚠️ Could not parse {best_file}: {e}")

# --- Compare and Save if Best ---
if avg_response_time < best_score:
    with open(best_file, "w") as f:
        f.write(f"{checkpoint_path},{avg_response_time}")
    print(f"🏆 New best model: {checkpoint_path} with avg response time {avg_response_time}")
else:
    print(f"📉 {checkpoint_path} scored {avg_response_time}, not better than best {best_score}")
