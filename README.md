# Modeling Renewable Energy Transitions as a Markov Decision Process:  
### A Q-Learning approach to minimizing fossil-fuel reliance while meeting energy demands  

**Authors**: Xander Hnasko, Luis Botin, and Jonathan Lee  
**Course**: CS238 Final Project  

---

## High-Level Overview

### Motivation
This project started from an interest in Germany's *Energiewende* policy initiative in 2016 — a commitment to fully transition the country to renewable energy by the end of the year. Energiewende aimed to eliminate reliance on fossil fuels, largely through the building and use of solar and wind infrastructure. Unfortunately, unprecedented weather patterns (less sun and wind than expected) meant that Germany was running a significant energy deficit, particularly in winter months. To meet demands, Germany was forced to reactivate old coal-fire plants it had shut down under the program. The upfront emissions-costs associated with reactivating old and inefficient coal plants are greater than the emissions-costs of simply leaving coal plants running. Thus, Germany's overall emissions for the year actually *increased*, despite the transition to renewable energy sources.  

Energiewende highlights the complexity of energy transition plans, as well as the stochastic nature of weather patterns and the inherent uncertainty in renewable energy production. This begs the question: "how can we minimize our reliance on fossil fuel energy while always meeting energy demands?"

### Goals
This project models our reliance on fossil fuels as a Markov Decision Process (MDP). Specifically focusing on wind power, we aim to optimize the proportion of renewable to fossil fuel usage in the short term. The objectives are to:
1. Minimize fossil fuel reliance while meeting energy demands.
2. Balance penalties for unmet demand with rewards for reduced fossil fuel reliance.

---

## Data
We are using the Techno-Economic Summary and Index dataset from the National Renewable Energy Laboratory. It contains summarized statistics for 120,000 wind plants within the continental United States. Each row in the dataset contains the location of the wind plant along with a variety of other information. For this project, we focus on:
- `wind_speed`
- `capacity_factor`

These variables help construct the state space for our MDP.

---

## Markov Decision Process (MDP)

### State Space
The state at time \(i\) is represented as:
$$
s_i = (\varphi_i, w_i, O(w_i), d_i)
$$

Where:
- \( \varphi_i \defeq \) Proportion of energy demand accounted for by fossil fuels at state \(i\), where \(0 \leq \varphi_i \leq 1\).
- \( w_i \defeq \) Wind speed at state \(i\).
- \( O(w_i) \defeq \) Power output of the wind turbines (MW):
$$
O(w_i) = 0.5 \cdot C_p \cdot \rho \cdot \pi \cdot R^2 \cdot w_i^3
$$
    - \(C_p\): Coefficient of performance (efficiency factor).  
    - \(\rho\): Air density (\(1.225 \, \text{kg/m}^3\)).  
    - \(R\): Blade radius (meters).  
    - \(w_i\): Wind speed (\( \text{m/s} \)).  
- \( d_i \defeq \) Energy demand in MW.

---

### Action Space
At each state, the agent chooses from the following actions:
$$
a_i \in \{\varphi_{(+)}, \varphi_{(-)}, \varphi_0\}
$$

Where:
- \( \varphi_{(+)} \): Increase \(\varphi_i\) by \(\Delta\), ensuring \(0 \leq \varphi_i + \Delta \leq 1\).
- \( \varphi_{(-)} \): Decrease \(\varphi_i\) by \(\Delta\), ensuring \(0 \leq \varphi_i - \Delta \leq 1\).
- \( \varphi_0 \): Leave \(\varphi_i\) unchanged.

---

### Transition Model

#### Wind Speed Transition \(P(w' | w)\)
The next wind speed is modeled as:
$$
P(w' | w) \sim \mathcal{N}(\mu_w', \sigma^2(w))
$$
Where:
$$
\mu_w' = \lambda \cdot w + (1 - \lambda) \cdot \mu_w
$$
- \(\lambda = e^{-\alpha \cdot t}\): Weighting factor that controls reversion to the historical mean \(\mu_w\).  
- \(\sigma^2(w)\): Variance of wind speed changes in the dataset.

---

### Reward Function
The reward function is defined as:
$$
R(s, a, s') =
\begin{cases} 
+10 \cdot (1 - \Delta \varphi), & \text{if demand is met.} \\
-50 + 10 \cdot (-\Delta \varphi), & \text{if demand is unmet.}
\end{cases}
$$

---

### Deriving the Optimal Policy
We solve the MDP using Q-learning. The action-value function \(Q(s, a)\) is updated as:
$$
Q(s, a) \gets Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
$$

### Optimal Policy and Q-Learning  

The optimal policy \( \pi^*(s) \) is derived as:  
\[
\pi^*(s) = \arg\max_a Q^*(s, a).
\]  

The Q-learning algorithm iteratively updates the \( Q \)-values based on observed transitions \((s, a, r, s')\) using the update rule:  
\[
Q(s, a) \gets Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right),
\]  

In the context of our renewable energy transition problem, the Q-learning algorithm iteratively updates \( Q(s, a) \) and derives the optimal policy \( \pi^*(s) \).  

As we iterate, we learn the best actions \( a \) to reduce fossil fuel reliance \( \varphi \) while also ensuring demand is met (\( O(w) + \varphi' \cdot d' \geq d' \)), taking into account stochastic transitions in wind speed \( w \) and energy demand \( d \). This allows us to optimize a reward structure that balances environmental goals with meeting energy demand.  
