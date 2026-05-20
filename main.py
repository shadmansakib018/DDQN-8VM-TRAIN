from flask_server import create_flask_app
from dqn_model import DQNAgent
import threading
import subprocess
import time

NUM_INSTANCES = 1
BASE_PORT = 6000
batch_size = 50
lb_type = 4
epochs = 2000
MODEL_PATH = "C:/Users/ss4587s/Desktop"
jar_path = "DDQN_TRAIN_8VM.jar"

agent = DQNAgent(MODEL_PATH+"/checkpoint_step_5200.pth")

fallback_app = create_flask_app(agent, BASE_PORT)

fallback_thread = threading.Thread(target=fallback_app.run, kwargs={"port": BASE_PORT})
fallback_thread.daemon = True
fallback_thread.start()
print(f"🛡️ Fallback server running on port {BASE_PORT}")
time.sleep(2)

port = str(BASE_PORT)
proc = subprocess.Popen(["java", "-jar", jar_path, port, str(batch_size), 
                         str(lb_type), str(epochs)])
time.sleep(1)


proc.wait()
