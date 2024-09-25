import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from q_network import QNetwork
from reply_buffer import ReplayBuffer

class DQAgent():
    def __init__(self, state_shape, action_space_size, BUFFER_SIZE,BATCH_SIZE,LR,GAMMA,TAU,UPDATE_EVERY, seed):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state_shape = state_shape
        self.action_space_size = action_space_size
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.lr = LR
        self.gamma = GAMMA
        self.tau = TAU
        self.update_every = UPDATE_EVERY

        self.seed = random.seed(seed)

        # Double Q-Network
        self.qnetwork_local = QNetwork(state_shape, action_space_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_shape, action_space_size, seed).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state,service_index, eps=0.):
        
        """Returns actions for given state as per current policy.
        Params
        ======
            state (our matrix): current state
            service_index (int): index of the column in the state matrix that represents the service
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.qnetwork_local.eval()  # Set to evaluation mode
        with torch.no_grad():
            q_values = self.qnetwork_local(state)  # Get Q-values for all actions (nodes)
        self.qnetwork_local.train()  # Set back to training mode
        
        # Extract the service presence for each node from the state matrix
        service_presence = state[:, :, service_index]  # service_index represents the column for requested service
              
        # Apply Q-masking by setting the Q-values for nodes without the service to a very negative value
        invalid_mask = (service_presence == 0).squeeze()  # Mask for nodes that don't have the requested service
        q_values = q_values.squeeze(0)  # Reshape to [4]
        q_values[invalid_mask] = float('-inf')  # Mask out invalid actions

        # Epsilon-greedy action selection
        if random.random() > eps:
            # Select the action with the highest Q-value among valid actions
            action = torch.argmax(q_values).item()
        else:
            # Randomly select a valid action (only from nodes that have the service)
            valid_actions = torch.where(service_presence.squeeze() == 1)[0]
            action = random.choice(valid_actions.cpu().numpy()).item()

        return action


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences
        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)

        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)

        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


