# Reinforcement Learning

Reinforcement learning (RL) studies how an agent learns to choose actions by interacting with an environment. The agent observes a state, takes an action, receives a reward, and transitions to a new state. Its goal is to maximize the expected sum of future rewards.

- [Markov Decision Process (MDP)](#markov-decision-process)
- [Value and Q-functions](#value-and-q-functions)
- [Bellman equations](#bellman-equations)
- [Continuous state spaces](#continuous-state-spaces)
- [Epsilon-greedy action selection](#epsilon-greedy-action-selection)
- [Deep Q-networks](#deep-q-networks)

**Gymnasium**

In practice, we separate the learning strategy from the environment, which makes training more general. A common framework is [Gymnasium](https://gymnasium.farama.org/) which provides a general API for environments to interact with agents.

## Markov Decision Process

An RL problem is commonly modeled as a Markov Decision Process (MDP), defined by:

- $S$: the set of states
- $A$: the set of actions
- $P(s' \mid s, a)$: the probability of reaching state $s'$ after taking action $a$ in state $s$
- $R(s, a, s')$: the reward for that transition
- $\gamma \in [0, 1]$: the discount factor, which controls the importance of future rewards

The Markov property means the next state depends only on the current state and action.

## Value and Q-functions

For a policy $\pi$, the state-value function $V^\pi(s)$ is the expected discounted return from state $s$ when following $\pi$. The action-value function, or Q-function, $Q^\pi(s, a)$, is the expected return from taking action $a$ in state $s$ and then following $\pi$:

$$
Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0=s, A_0=a\right]
$$

The expectation accounts for randomness in rewards and state transitions. If transition probabilities are known, it can be computed as a probability-weighted sum rather than by sampling outcomes.

## Bellman equations

The Bellman expectation equations relate values to the immediate reward and the value of the next state.
For the optimal action-value function, the Bellman optimality equation is:

$$
Q^*(s, a) = \sum_{s'} P(s' \mid s, a)
            \left[R(s, a, s') + \gamma \max_{a'} Q^*(s', a')\right]
$$

## Continuous state spaces

A state space is continuous when a state is described by real-valued measurements, such as a robot's position and velocity. There may be infinitely many possible states, so storing a separate value for every state-action pair in a table is not practical.

A continuous state space does not necessarily mean the actions are continuous. Tabular Q-learning and DQN can use continuous states with a finite set of actions when a function approximator represents $Q(s, a)$. Standard DQN selects among discrete actions and reflect them to continuous spaces.

## Epsilon-greedy action selection

An $\varepsilon$-greedy policy balances exploration with current state. It chooses a random action with probability $\varepsilon$ and otherwise chooses a greedy action:

$$
a_t =
\begin{cases}
\text{a random action}, & \text{with probability } \varepsilon \\
\arg\max Q(s, a), & \text{with probability } 1-\varepsilon
\end{cases}
$$

Random exploration may occasionally select a greedy action as well. A common training schedule starts with a relatively high $\varepsilon$ and gradually reduces it to a small floor allowing broad exploration early. The schedule and floor are environment dependent. Exploration is often reduced or disabled during evaluation.

## Deep Q-networks

A Deep Q-Network (DQN) uses a neural network to approximate action values, $Q(s, a; \theta)$, making Q-learning practical when the state space is too large for a table. For each state, the network typically outputs one value per discrete action, and the agent selects actions using an exploration policy such as $\varepsilon$-greedy.

DQN stabilizes learning with two techniques:

- **Experience replay:** store transitions $(s, a, r, s', \text{done})$ in a replay buffer and train on randomly sampled mini-batches. This reduces correlation between consecutive observations and allows transitions to be reused.
- **Target network:** maintain a second network with parameters $\theta^{-}$ to compute training targets. Periodically copy the online network parameters $\theta$ to the target network, or update them gradually.

For a sampled transition, the target is

$$
y =
\begin{cases}
r, & \text{if the transition is terminal} \\
r + \gamma \max_{a'} Q(s', a'; \theta^-), & \text{otherwise}
\end{cases}
$$

The online network is trained to reduce the temporal-difference error, commonly with the squared loss

$$
L(\theta) = \left(y - Q(s, a; \theta)\right)^2
$$

The target network is not updated by this loss; it is synchronized on its own schedule. DQN combines function approximation with the Q-learning update, but can still be sensitive to hyperparameters and is intended for discrete action spaces.
