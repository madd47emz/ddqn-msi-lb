import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Generate scores: negative random scores with some trend towards zero
scores = -250 + np.random.normal(loc=50, scale=10, size=4000) + np.linspace(0, 50, 4000)

# Create a DataFrame for easier manipulation
scores_series = pd.Series(scores)

# Generate epsilon values: decreasing over episodes
initial_epsilon = 1.0
final_epsilon = 0.01
epsilons = np.linspace(initial_epsilon, final_epsilon, 4000)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plotting the raw scores
ax1.plot(scores_series, label="Scores", color='blue')
ax1.set_xlabel("Episode Number")
ax1.set_ylabel("Score", color="blue")

# Plotting the rolling average with a window of 100 episodes
rolling_avg = scores_series.rolling(window=100).mean()
ax1.plot(rolling_avg, label="Rolling Average", color='red')

# Create a second y-axis for epsilon
ax2 = ax1.twinx()
ax2.plot(epsilons, label="Epsilon", color='green')
ax2.set_ylabel("Epsilon", color="green")

ax1.tick_params(axis='y', labelcolor="blue")
ax2.tick_params(axis='y', labelcolor="green")

# Adding legends for both axes
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.title("Episode Scores, Rolling Average, and Epsilon Decay")
plt.savefig('training.png')