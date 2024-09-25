import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from hyper_param import BATCH_SIZE, BUFFER_SIZE, GAMMA, LR, TAU, UPDATE_EVERY
from gym_env import CustomK8sEnv
from dqn_agent import DQAgent

app = FastAPI()
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
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

# Endpoint to redirect based on the service name
@app.get("/redirect-to-service")
async def redirect_to_service(service_name: str):
    try:
        ip, port = predict_service_address(service_name) 
        redirect_url = f"http://{ip}:{port}/api/test"
        return RedirectResponse(url=redirect_url)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

