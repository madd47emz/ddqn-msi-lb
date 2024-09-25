import time
import gym
from gym import spaces
import numpy as np
import requests

from nodes_metric import get_node_metrics
from list_ms import get_services_with_node_ports
from msi_placement import get_pod_placements_for_service
from msi_metric import get_pod_cpu_usage_for_service, get_pod_ram_usage_for_service


class CustomK8sEnv(gym.Env):
    def __init__(self):
        super(CustomK8sEnv, self).__init__()
        
        self.services_info = get_services_with_node_ports()
        self.nodes_info = get_node_metrics()

        # Action Decoding: Each action corresponds to a node-service pair
        self.node_ips = [node_data['node_ip'] for node_data in self.nodes_info.values()]
        self.service_ports = [service['node_ports'] for service in self.services_info.values()]

        # Action space: Discrete space to select a node
        self.action_space = spaces.Discrete(len(self.node_ips))

        # Observation space: We observe node CPU and memory usage percentages and service placements
        self.observation_space = spaces.Box(
            low=0, 
            high=100, 
            shape=(len(self.node_ips), 2 + len(self.services_info)),
            dtype=np.float32
        )

    def _get_msi_placement(self, app_label):
        """
        Get the current pod placement for the given app label (microservice).
        Returns a binary victor indicating which nodes are running the service pods.
        """
        pod_placements = get_pod_placements_for_service(app_label)
        placement = np.zeros(len(self.node_ips))

        for pod,details in pod_placements.items():
            if details['status'] == 'Running':
                node_index = self.node_ips.index(self.nodes_info[details['node']]['node_ip'])
                placement[node_index] = 1 

        return placement
    
    def _get_state(self):
        self.nodes_info = get_node_metrics()
        state = []

        # Initializing the state with just the CPU and memory data for each node
        for node_name, node_data in self.nodes_info.items():
            node_state = [node_data['cpu_percentage'], node_data['memory_percentage']]
            state.append(node_state)
        state = np.array(state)

        # Iteration over the services and add the placement arrays vertically
        for service_name, service_info in self.services_info.items():
            app_label = service_info['app_label']
            placement = self._get_msi_placement(app_label)

            # Converting the placement to a column vector and append it to the state
            placement_column = np.array(placement).reshape(-1, 1)  # Reshape to be a column
            state = np.hstack((state, placement_column))  # Horizontally stack the placement

        return state
    
    def _calculate_imbalance(self):
        """
        Calculate the imbalance metric for load balancing.
        referring to the paper : https://qspace.qu.edu.qa/handle/10576/55032?show=full
        """
        total_imbalance = 0
        num_microservices = len(self.services_info)

        for service_name, service_info in self.services_info.items():
            app_label = service_info['app_label']

            # Get utilization (CPU) data for all instances of this microservice
            msi_cpu_usage = get_pod_cpu_usage_for_service(app_label)

            # Get utilization (RAM) data for all instances of this microservice
            msi_ram_usage = get_pod_ram_usage_for_service(app_label)

            combined_array = [list(pair) for pair in zip(msi_cpu_usage, msi_ram_usage)]

            # Calculate CPU imbalance for this microservice (standard deviation of CPU and RAM usage)
            imbalance = np.sqrt(np.mean([(cpu_usage - np.mean(msi_cpu_usage)) ** 2 + (ram_usage - np.mean(msi_ram_usage)) ** 2 for cpu_usage, ram_usage in combined_array]))

            # Add to total imbalance
            total_imbalance += imbalance

        # Average imbalance across all microservices
        avg_imbalance = total_imbalance / num_microservices
        return avg_imbalance

    def step(self, action, requested_service):
        

        #Decoding (action, requested_service) to (node_ip, node_port)
        chosen_node_ip = self.node_ips[action]
        chosen_node_port = self.service_ports[requested_service]

        # Build action response: <node-ip>:<node-port>
        action_result = f"{chosen_node_ip}:{chosen_node_port}"

        # API call to the selected node and service
        api_url = f"http://{action_result}/api/test"
        retries = 3
        backoff_factor = 2  # Exponential backoff factor

        for attempt in range(retries):
            try:
            # Attempt API call with a timeout
                response = requests.get(api_url, timeout=40)

            # Check if the request was successful
                if response.status_code == 200:
                    print(f"API call successful: {api_url}")
                    break
                else:
                    print(f"API call failed with status {response.status_code}: {api_url}")

            except requests.exceptions.RequestException as e:
                print(f"Error during API call: {e}. Attempt {attempt + 1} of {retries}")
            # If not the last attempt, wait for some time before retrying
            if attempt < retries - 1:
                sleep_time = backoff_factor ** attempt  # Exponential backoff
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to reach the service after {retries} attempts.")


        # Fetch updated state
        self.state = self._get_state()

        # Calculate reward based on imbalance metric
        reward = -self._calculate_imbalance()

        # Checking for termination condition (reached a threshold)
        done = False
        penalty = 0
        for node_state in self.state:
        # If CPU or RAM usage exceeds 90%, add a penalty
            if node_state[0] > 90 or node_state[1] > 90:
                penalty += 1

        #penalize the agent with number of nodes that have exceeded the threshold
        if penalty > 0:
            done = True
            reward = -penalty

        return self.state, reward, done, {'action_result': action_result}

    def reset(self):
        self.state = self._get_state()
        return self.state

    def render(self, mode='human'):
        for node_ip, metrics in self.nodes_info.items():
            print(f"Node {node_ip}: CPU Usage = {metrics['cpu_percentage']}%, Memory Usage = {metrics['memory_percentage']}%")

# if __name__ == "__main__":
#     services = get_services_with_node_ports()
#     for service_name, service_info in services.items():
#         print(f"Service {service_name}")
#         print(env.get_msi_placement(service_info['app_label']))

    