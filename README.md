#Middleware

In this research, we developed a custom middleware that replaces the traditional Kubernetes ingress controller and acts as an API gateway. This middleware exposes a single
endpoint where the name of the requested service is provided as a query_parameter. Upon receiving a request, the middleware performs the following operations:

1. **Model Loading for Evaluation:** The middleware dynamically loads the deep learning model for evaluation using torch.load from server-stored checkpoint files checkpoint.pth.
2. **State Management and Decision Making:** The current state of the system isobtained by invoking the get_state method from CustomK8sEnv.
3. **Action Decoding:** The middleware decodes the action using a logic similar to that defined in the step method of the CustomK8sEnv class.
4. **Request Forwarding:** Once decoded, the middleware forwards the API request to the node IP associated with the action, appending the requested service to the endpoint (i.e., <nodeIp>:requestedService).

#Middleware Kubernetes Use Cases

1. The middleware replaces the Ingress Controller (e.g., Nginx Ingress) to forward traﬀic to the appropriate Kubernetes service.
2. It replaces the Kubernetes Service Discovery by listing the services, their pod placements in the worker nodes, and their status (e.g., Running, Pending, etc.) to feed them to the DDQN model, helping to inform load-balancing decisions.
3. It fetches Microservice Instance Placement to feed them to the DDQN model, helping to inform load-balancing decisions.
4. Similarly, it fetches Node Metrics (CPU% and RAM%) to provide a broader view of resource usage across the cluster, which is crucial for the DDQN model.


#DDQN Load Balancer:

The core component of this architecture is a DDQN model. This model is trained to select the most suitable worker node for handling traﬀic, considering the current systemload and resource usage.

1. **Observation Space:** The DDQN model receives real-time observations from the Kubernetes environment, including:
  • CPU and RAM usage of each worker node.
  • Current microservice instance placements across the worker nodes.

  The observation space includes CPU and memory usage for each node, along with a binary vector indicating the presence of each microservice on the node. If there are N nodes and K microservices, the observation space can be represented as matrix of **N × (2 + K)**.

2. **Action Space (Traﬀic Routing Decisions):** This action directs traﬀic to a specific worker node hosting requested service based on the current system state.The action space is defined as selecting a node. If you have N nodes, the total number of actions is represented as N.

3. **Reward:** is designed to minimize imbalance while The reward at time step t, denoted as Rt = -Im.
     DDQN model will be penalized For every worker node that reaches a predefined CPU or RAM usage threshold, a penalty of −1 will be added to the reward.

4. **Q-Masking:** Creates a mask to filter out invalid actions based on service presence using service_index to pin in state matrix placement victor. if the requested service is not present, the q value will set to minus infinity.

5. Uses epsilon-greedy strategy to select an action.

6. Calculates the loss using Mean Squared Error (MSE) between expected and target Q-values.

7. Performs a gradient descent step to minimize the loss.
  
9. Updates the target network using a soft update.

#Metrics Collection

The environment is designed for reinforcement learning with Kubernetes. It interacts with the k8s-metric-server of the cluster Using kubernetes.client Python library. At each service request, the environment collects real-time metrics from the Kubernetes cluster using various API calls. These include:

1. **Nodes Capacity:** Name, IP address, CPU capacity(millicores), RAM capacity(megabytes) for each node, retrieved through the get_node_capacity() function.
2. **Nodes Usage:** Name, IP address, normalized(percentage) CPU and RAM usage for each node, retrieved through the get_node_metrics() function with the help of get_node_capacity() function for normalization.
3. **Service Discovery:** Name, NodePort, App Label, for each service using the get_services_with_node_ports() function.
4. **Pods Discovery:** Pod Name Node Name, Pod Status, for each pod of the service via the get_pod_placements_for_service(app_label) function.
5. **Microservice Instance Usage:** normalized pod usage, for each service via the get_pod_cpu_usage_for_service(app_label) function and get_pod_ram_usage_for_service(app_label) function with help of get_node_capacity() function for normalization.


#Replay Memory

We crafted ReplayBuffer Class - a cyclic buffer of bounded size that holds the transitions observed recently.

1. self.experience - a named tuple representing a single experience in our environment. It essentially contain (state, action, reward, next_state, done).
2. self.memory - a double-ended queue with bounded size.
3. add(state, action, reward, next_state, done) - a function that append experience tuple to memory.
4. sample() - a function for selecting a random batch of experiences for training.


#Q-network

We crafted QNetwork class that extends torch.nn.Module(neural network). it overrides forward function that takes in the difference between the current and previous cluster state. Since the network layers are of type Linear, we perform state.flatten for the input layer. hidden layers are 64 of length. It has action_space_size as outputs.

#Hyper-params

• BUFFER_SIZE=int(1e5) #replay buffer size
• BATCH_SIZE=32 #sample minibatch size
• GAMMA=0.99 #discount factor
• TAU=1e-3 #for soft update of target network parameters
• LR=5e-4 #learning rate
• UPDATE_EVERY=10 #how often to update the network

#Hyperparameter Optimization

For our DQN model, Optuna was utilized to tune the hyperparameters.
