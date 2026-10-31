import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


MAP_NAME = "4x4"
EPISODES = 3000
EVALUATION_EPISODES = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
SEED = 42


def train_agent():
    env = gym.make("FrozenLake-v1", map_name=MAP_NAME, is_slippery=False)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rng = np.random.default_rng(SEED)
    env.action_space.seed(SEED)
    epsilon = EPSILON_START
    episode_rewards = []

    try:
        for episode in range(EPISODES):
            state, _ = env.reset(seed=SEED if episode == 0 else None)
            total_reward = 0
            terminated = truncated = False

            while not (terminated or truncated):
                if rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(q_table[state]))

                next_state, reward, terminated, truncated, _ = env.step(action)

                # Update the state-action value immediately from this transition.
                future_value = 0 if terminated else np.max(q_table[next_state])
                target = reward + DISCOUNT_FACTOR * future_value
                q_table[state, action] += LEARNING_RATE * (
                    target - q_table[state, action]
                )

                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        map_description = env.unwrapped.desc.copy()
    finally:
        env.close()

    return q_table, episode_rewards, map_description


def evaluate_agent(q_table):
    env = gym.make("FrozenLake-v1", map_name=MAP_NAME, is_slippery=False)
    successes = 0

    try:
        for _ in range(EVALUATION_EPISODES):
            state, _ = env.reset()
            terminated = truncated = False

            while not (terminated or truncated):
                action = int(np.argmax(q_table[state]))
                state, reward, terminated, truncated, _ = env.step(action)

            successes += reward
    finally:
        env.close()

    return successes


def plot_policy(q_table, map_description):
    arrows = ["←", "↓", "→", "↑"]
    rows, cols = map_description.shape
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks(np.arange(cols + 1))
    ax.set_yticks(np.arange(rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="black", linewidth=2)
    ax.set_aspect("equal")

    for state in range(rows * cols):
        row, col = divmod(state, cols)
        tile = map_description[row, col].decode()

        if tile in ("H", "G"):
            label = tile
        else:
            label = arrows[int(np.argmax(q_table[state]))]
            if tile == "S":
                label = f"S\n{label}"

        ax.text(col + 0.5, row + 0.5, label, ha="center", va="center", fontsize=22)

    ax.set_title("FrozenLake 4x4: learned greedy policy")
    plt.show()


def main():
    q_table, episode_rewards, map_description = train_agent()
    successes = evaluate_agent(q_table)

    print(f"Training episodes: {EPISODES}")
    print(
        "Successes in final 100 training episodes: "
        f"{sum(episode_rewards[-100:])}/100"
    )
    print(
        f"Greedy policy test success rate: "
        f"{successes}/{EVALUATION_EPISODES}"
    )

    plot_policy(q_table, map_description)


if __name__ == "__main__":
    main()
