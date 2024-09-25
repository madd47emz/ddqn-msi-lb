import random
from gym_env import CustomK8sEnv
from dqn_agent import DQAgent
import torch

from hyper_param import BATCH_SIZE, BUFFER_SIZE, GAMMA, LR, TAU, UPDATE_EVERY


env = CustomK8sEnv()

# Placeholder function to simulate model loading
def load_model():
    agent = DQAgent(state_shape=env.observation_space.shape, action_space_size=env.action_space.n,BUFFER_SIZE = int(BUFFER_SIZE), BATCH_SIZE = BATCH_SIZE,LR = LR,GAMMA = GAMMA,TAU = TAU,UPDATE_EVERY = UPDATE_EVERY, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint1.pth', weights_only=True))
    agent.qnetwork_local.eval()
    return agent  
model = load_model()

# Function to simulate AI prediction (replace this with actual model logic later)
def predict_service_address(service_name: str):
    print(env.services_info[service_name]["node_ports"])
    try:
        service_index = env.service_ports.index(env.services_info[service_name]["node_ports"])
    except ValueError:
        service_index = -1
    if service_index != -1:
        state = env.reset()
        action = model.act(state,service_index+2, 0)
        service_port = env.service_ports[service_index]
        node_ip = env.node_ips[action]
        return node_ip, service_port 
    else:
        return "Service not found"
    
print(predict_service_address("ms-1-srv"))
