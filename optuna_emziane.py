import random
import optuna
from dqn_agent import DQAgent
from gym_env import CustomK8sEnv
import matplotlib.pyplot as plt

def objective(trial,eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    # Suggest hyperparameters
    BUFFER_SIZE = trial.suggest_float('BUFFER_SIZE', 1e4, 1e6, log = True)  # Between 10k and 1M experiences
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [32, 64, 128, 256])  # Try different batch sizes
    GAMMA = trial.suggest_uniform('GAMMA', 0.9, 0.999)  # Discount factor between 0.9 and 0.999
    TAU = trial.suggest_loguniform('TAU', 1e-4, 1e-2, log = True)  # Soft update parameter
    LR = trial.suggest_loguniform('LR', 1e-5, 1e-3, log = True)  # Learning rate between 1e-5 and 1e-3
    UPDATE_EVERY = trial.suggest_int('UPDATE_EVERY', 4, 10)  # How often to update the network

    # Create and train your DQN model with the suggested hyperparameters
    env = CustomK8sEnv()
    agent = DQAgent(state_shape=env.observation_space.shape, action_space_size=env.action_space.n,BUFFER_SIZE = int(BUFFER_SIZE), BATCH_SIZE = BATCH_SIZE,LR = LR,GAMMA = GAMMA,TAU = TAU,UPDATE_EVERY = UPDATE_EVERY, seed=0)
    
    # Keep track of cumulative reward over all episodes
    total_reward = 0
    eps = eps_start 
    for episode in range(100): 
        state = env.reset()
        rows, columns = state.shape
        score = 0
        for t in range(100):
            requested_service = random.randint(2, columns-1)
            action = agent.act(state,requested_service, eps)
            next_state, reward, done, _ = env.step(action,requested_service-2)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        eps = max(eps_end, eps_decay*eps)
        total_reward += score
        
        # Report the intermediate result to Optuna (after each episode)
        trial.report(total_reward, episode)

        # Check whether the trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return -total_reward

# Optimize the objective function using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

# Plotting the optimization history
fig, ax = plt.subplots(figsize=(10,5))
optuna.visualization.plot_optimization_history(study)
ax.set_title("Optuna Optimization History")
ax.set_xlabel("Number of Trials")
ax.set_ylabel("Objective Value (Cumulative Reward)")
plt.savefig('optuna.png')

# Get the best hyperparameters after optimization
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

