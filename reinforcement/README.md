# Reinforcement Learning

Reinforcement learning (RL) studies how an agent learns to choose actions by interacting with an environment. The agent observes a state, takes an action, receives a reward, and transitions to a new state. Its goal is to maximize the expected sum of future rewards.

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

