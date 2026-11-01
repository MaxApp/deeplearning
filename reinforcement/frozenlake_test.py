import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_size, action_size))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Q-Learning
        td_error = target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_manhattan_distance(state, goal_state, grid_size=8):
    x1, y1 = divmod(state, grid_size)
    x2, y2 = divmod(goal_state, grid_size)
    return abs(x1 - x2) + abs(y1 - y2)

def train_frozenlake(map_name="8x8", episodes=2000, is_slippery=False):
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=None)
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    agent = QLearningAgent(
        state_size, 
        action_size, 
        learning_rate=0.1, 
        gamma=0.95, 
        epsilon=1.0, 
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    rewards_history = []
    success_count = 0
    
    # 8x8 last state (63)
    goal_state = state_size - 1 
    grid_size = 8 if map_name == "8x8" else 4 if map_name == "4x4" else 16
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # --- Reward Shaping ---
            # instead of using only 0/1
            # we add minor rewards when target is closer
            shaped_reward = reward
            if reward == 0 and not done:
                old_dist = get_manhattan_distance(state, goal_state, grid_size)
                new_dist = get_manhattan_distance(next_state, goal_state, grid_size)
                if new_dist < old_dist:
                    shaped_reward = 0.01
                else:
                    shaped_reward = -0.01
            
            agent.learn(state, action, shaped_reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if reward == 1:
                success_count += 1
        
        # decay epsilon each epoch
        agent.decay_epsilon()
                
        rewards_history.append(total_reward)
        
        if (e + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode: {e+1}/{episodes}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    env.close()
    return agent, rewards_history, success_count

def visualize_training(rewards_history, window_size=100):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label='Reward per Episode', alpha=0.3)
    
    if len(rewards_history) >= window_size:
        cumsum = np.cumsum(np.insert(rewards_history, 0, 0)) 
        moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        plt.plot(range(window_size-1, len(rewards_history)), moving_avg, label=f'Moving Average ({window_size})', color='red')
        
    plt.title('FrozenLake Q-Learning Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    MAP_NAME = "8x8" 
    EPISODES = 2000
    IS_SLIPPERY = False

    agent, rewards, successes = train_frozenlake(map_name=MAP_NAME, episodes=EPISODES, is_slippery=IS_SLIPPERY)

    print(f"success: {successes}/{EPISODES}")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward per Episode', alpha=0.3)

    window = 100
    if len(rewards) >= window:
        cumsum = np.cumsum(np.insert(rewards, 0, 0)) 
        moving_avg = (cumsum[window:] - cumsum[:-window]) / window
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Moving Average ({window})', color='red')

    plt.title('FrozenLake 8x8 Q-Learning Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

