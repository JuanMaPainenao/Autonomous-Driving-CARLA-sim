# Reinforcement Learning

## 1. What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an agent learns by interacting with an environment. There's no labeled dataset — the agent takes actions, gets rewards (or penalties), and over time figures out which actions lead to the best outcomes [1].

The basic loop is:

$$
Agent \xrightarrow{action} Environment \xrightarrow{state, reward} Agent
$$

This is different from supervised learning (where you have input-output pairs) and unsupervised learning (where you find patterns in data). In RL, the agent has to *discover* what works through trial and error.

## 2. Markov Decision Process (MDP)

An MDP is the mathematical framework behind RL. It's defined by a tuple $(S, A, P, R, \gamma)$:

- $S$ — set of possible states
- $A$ — set of possible actions
- $P(s'|s, a)$ — probability of transitioning to state $s'$ after taking action $a$ in state $s$
- $R(s, a, s')$ — reward received after that transition
- $\gamma \in [0, 1]$ — discount factor (how much we care about future rewards)

The **Markov property** says: the next state depends only on the current state and action, not on the entire history. This simplification is what makes the math tractable [1].

The agent's goal is to find a **policy** $\pi(a|s)$ that maximizes the expected **return** (cumulative discounted reward):

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

## 3. Value Functions and Bellman Equation

The **state-value function** tells us how good it is to be in state $s$ following policy $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]
$$

The **action-value function** tells us how good it is to take action $a$ in state $s$:

$$
Q^\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]
$$

Both satisfy **Bellman equations** — recursive relationships that express a state's value in terms of its successors' values. This recursion is the foundation of dynamic programming and TD learning [1].

The **advantage function** measures how much better a specific action is compared to the average:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

If $A > 0$, the action is better than average. If $A < 0$, it's worse. This is key for policy gradient methods [3].

## 4. Policy Gradient Methods

Instead of learning a value function and deriving a policy from it, policy gradient methods directly optimize the policy parameters $\theta$. The **policy gradient theorem** gives us the gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s, a) \right]
$$

Intuitively: increase the probability of actions with positive advantage, decrease the probability of actions with negative advantage [1].

The problem is that these gradient estimates can be very noisy (high variance), which makes training unstable.

## 5. Proximal Policy Optimization (PPO)

PPO [4] is one of the most popular RL algorithms. It's a policy gradient method that solves a critical problem: if you update the policy too aggressively, performance can collapse.

### 5.1 The Problem

Standard policy gradients don't limit *how much* the policy changes per update. One bad update can ruin everything. TRPO [5] solved this by constraining the KL divergence between old and new policies, but it required expensive second-order optimization. PPO gets similar stability with simple first-order methods.

### 5.2 Clipped Surrogate Objective

PPO defines the probability ratio between new and old policies:

$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}
$$

Then it clips this ratio to prevent large changes:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \cdot \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot \hat{A}_t \right) \right]
$$

The `clip` function keeps $r_t$ in $[1-\varepsilon, 1+\varepsilon]$ (typically $\varepsilon = 0.2$). The `min` picks the more conservative estimate. This way, the policy can't change too much in one step [4].

### 5.3 Multiple Epochs

Unlike standard policy gradient (one gradient step per batch), PPO reuses the same batch for multiple epochs. The clipping ensures this reuse doesn't destabilize training. This makes PPO much more sample-efficient [4].

### 5.4 Generalized Advantage Estimation (GAE)

PPO uses GAE [3] to estimate the advantage function. It introduces a parameter $\lambda$ that controls the bias-variance tradeoff:

$$
\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

- $\lambda = 0$ → one-step TD (low variance, high bias)
- $\lambda = 1$ → Monte Carlo (high variance, low bias)
- $\lambda = 0.95$ → good practical balance

## 6. Neural Networks as Function Approximators

When the state space is huge (like images), we can't store a value for every state. We use neural networks to approximate $\pi_\theta$ and $V_\phi$ [2].

For **image observations**, CNNs extract spatial features (edges, shapes, objects) from raw pixels. For **mixed observations** (image + numerical data like speed or sensor readings), a **multi-input architecture** processes each modality separately — CNN for images, MLP for vectors — and concatenates them before the policy/value heads. This gives the agent explicit access to information that would be very hard to extract from pixels alone.

## 7. Key Hyperparameters

| Parameter | What it does |
|---|---|
| `learning_rate` | Step size for gradient updates |
| `n_steps` | Steps collected before each policy update |
| `batch_size` | Minibatch size within each epoch |
| `n_epochs` | Passes over collected data per update |
| `gamma` ($\gamma$) | Discount factor — how much future rewards matter |
| `clip_range` ($\varepsilon$) | How much the policy can change per update |
| `ent_coef` | Entropy bonus — encourages exploration |
| `gae_lambda` ($\lambda$) | Bias-variance tradeoff in advantage estimation |
| `target_kl` | Early stop threshold if policy changes too much |

## 8. Reward Shaping

The reward function is arguably the most important design choice in applied RL. The agent will optimize *whatever* signal it gets, which doesn't always match what we want. Common issues:

- **Reward hacking** — the agent finds loopholes (e.g., staying still to avoid crash penalties instead of learning to drive)
- **Sparse rewards** — if feedback is too rare, the agent can't learn early on
- **Scale imbalance** — if penalties are way larger than positive rewards, the agent focuses on avoidance instead of doing the actual task

Good reward functions decompose the goal into components (speed, lane centering, progress, collision avoidance) and keep their magnitudes balanced [1].

## 9. On-Policy vs Off-Policy

**On-policy** (PPO, A2C): learns from data collected by the *current* policy. Old data is discarded after each update. Simpler and more stable, but less sample-efficient.

**Off-policy** (SAC, DQN): learns from data collected by *any* policy, stored in a replay buffer. More sample-efficient, but harder to stabilize [1].

PPO compensates for its on-policy nature by doing multiple epochs per batch, squeezing more learning out of each data collection phase.

## References

[1] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA: MIT Press, 2018.

[2] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, 2015.

[3] J. Schulman, P. Moritz, S. Levine, M. I. Jordan, and P. Abbeel, "High-dimensional continuous control using generalized advantage estimation," in *Proc. ICLR*, 2016. arXiv:1506.02438.

[4] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.

[5] J. Schulman, S. Levine, P. Abbeel, M. I. Jordan, and P. Moritz, "Trust region policy optimization," in *Proc. ICML*, 2015. arXiv:1502.05477.
