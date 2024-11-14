# Renewable-Grid-Optimization


**Goal**
The objective is to develop an optimization strategy for managing energy storage systems within a smart grid. The goal is to maximize grid stability and minimize operational costs by dynamically adjusting storage usage in response to demand (real time), renewable energy supply fluctuations and grid constraints. Here, success would be defined by the creation of a model that can effectively reduce grid strain during peak hours and leverage renewable energy generation while also ensuring storage availability during demand surges. There exist many challenges in integrating renewable energy sources with efficient storage solutions, and this will become increasingly critical given the rising share of renewables in the energy mix – we are curious to see how sequential decision-making models play into this, ensuring the efficiency/resilience of such systems.
**Decision Making**
Core decision making problem is determining optimal actions for charging/discharging or holding energy in storage systems over some time horizon. Sequential decision-making comes into play here as storage usage must adapt dynamically to forecasted and real time data on demand + supply. The environment comprises renewable energy sources (solar, wind), grid energy demand and storage systems. Decisions made at each time step affect the state of the grid (eg storage levels, demand fulfillment), which means there’s a need to manage actions sequentially.
**We model it as an MDP:**
 - **States:** Current storage levels, forecasted demand, renewable generation forecasts, and grid congestion levels (additional state info may include historical demand patterns or weather conditions – these impact renewable generation)
- **Actions:** Actions are charge, discharge, or hold decisions at each time step. A charge action will fill the storage from available generation, discharge will supply stored energy to grid, and hold actions maintain current storage level. The goal is to make the best choice at each step to prepare for anticipated changes in demand/supply.
- **Transitions:** The transition dynamics model changes in the environment, such as fluctuations in demand or in renewable generation. This could involve stochastic models for demand forecasting or renewable generation variability – this captures uncertainties inherent in the system.
- **Rewards:** Reward function will balance several factors: cost savings from reduced grid purchases, revenues from discharging storage during peak prices, as well as penalties associated with unmet demand / energy shortages. We max this reward over time as model learns an optimal policy for storage management.

**Uncertainty**
Sources of uncertainty would include:
- Demand Fluctuations
- Predicting demand, especially peak and off peak times, can be challenging and is affected by external factors (eg weather, human activity patterns).
- Renewable Generation Variability
- Inherent variability in renewable energy sources such as solar/wind (these are highly dependent on weather conditions which creates unpredictable variations in supply)
- Market Price Volatility
- Fluctuations in energy prices can fluctuate which may also influence when to use stored energy vs purchasing from the grid
