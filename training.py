import random
from gym_env import CustomK8sEnv
from dqn_agent import DQAgent
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import torch
import pandas as pd

from hyper_param import BATCH_SIZE, BUFFER_SIZE, GAMMA, LR, TAU, UPDATE_EVERY

env = CustomK8sEnv()

def dqn(agent, n_episodes=100, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=50)  # last 10 scores
    eps = eps_start                    # initialize epsilon
    epsilons = []                      # list containing epsilon values

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        rows, columns = state.shape
        score = 0
        for t in range(max_t):
            requested_service = random.randint(2, columns-1)
            action = agent.act(state,requested_service, eps)
            next_state, reward, done, _ = env.step(action,requested_service-2)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        epsilons.append(eps)              # save epsilon value

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores, epsilons


DQagent = DQAgent(state_shape=env.observation_space.shape, action_space_size=env.action_space.n,BUFFER_SIZE = BUFFER_SIZE, BATCH_SIZE = BATCH_SIZE,LR = LR,GAMMA = GAMMA,TAU = TAU,UPDATE_EVERY = UPDATE_EVERY, seed=0)
scores, epsilons = dqn(DQagent)
scores_series = pd.Series(scores)
epsilons = np.array(epsilons)  # Convert to NumPy array if necessary

fig, ax1 = plt.subplots(figsize=(10,5))

# Plotting the raw scores
ax1.plot(scores_series, label="Scores", color='blue')
ax1.set_xlabel("Episode Number")
ax1.set_ylabel("Score", color="blue")

# Plotting the rolling average with a window of 15 episodes
rolling_avg = scores_series.rolling(window=10).mean()
ax1.plot(rolling_avg, label="Rolling Average", color='orange')

# Create a second y-axis for epsilon
ax2 = ax1.twinx()  
ax2.plot(epsilons, label="Epsilon", color='red')
ax2.set_ylabel("Epsilon", color="red")

ax1.tick_params(axis='y', labelcolor="blue")
ax2.tick_params(axis='y', labelcolor="red")

# Adding legends for both axes
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.title("Episode Scores, Rolling Average, and Epsilon Decay")
plt.savefig('training.png')
