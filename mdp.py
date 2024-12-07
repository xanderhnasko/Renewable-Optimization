from packages import *
from data import get_data   
from visualizations import *
CP = 0.32

def get_data(filepath="data.pkl"):
    if not os.path.exists(filepath):
        raise FileNotFoundError("Data file not found. Ensure the data has been downloaded and saved correctly.")
    return pd.read_pickle(filepath)


def calculate_power_output(wind_speed, cp = CP, rho=1.225, R=130.5):
    # Cp: Average Coefficient of Performance across all wind turbines
    # rho: U.S Average wind density
    # R: Average Rotor Radius of modern tubines (2023+)
    return round((0.5 * cp * rho * np.pi * R**2 * wind_speed**3)/1e6, 3)


def wind_speed_transition(current_wind_speed, mu_w, sigma_w, alpha=0.1, t=1):
    lambda_ = np.exp(-alpha * t) 
    mu_w_prime = lambda_ * current_wind_speed + (1 - lambda_) * mu_w
    return max(np.random.normal(mu_w_prime, sigma_w), 0)

# this is placeholder code for now 
# we also need something for mu_d, sigma_d
def energy_demand_transition(current_demand, mu_d, sigma_d, alpha=0.1, t=1):
    lambda_ = np.exp(-alpha * t)
    mu_d_prime = lambda_ * current_demand + (1 - lambda_) * mu_d
    return max(np.random.normal(mu_d_prime, sigma_d), 0.1)

# TBU Fix this, JLEE - where do we use this? is it necessary?
def simulate_initial_wind_speed(n, mu_w=7.58, sigma_w=1.02):
    # mu_w and sigma_w derived from overall dataset statistics
    return np.random.normal(mu_w, sigma_w, n)
                  
# ACTION FUNCTIONS

def apply_action(row, action, delta=0.025):
    # delta is how much we increase or decrease the proportion of energy met by fossil fuels
    phi = row[0]
    if action == 'increase':
        phi = round(min(phi + delta, 1), 3)
    elif action == 'decrease':
        phi = round(max(phi - delta, 0), 3)
    # 'no_change' action does nothing
    return phi

def full_transition(state, action, mu_w, sigma_w, mu_d, sigma_d, alpha=0.1, t=1):
    """Compute the full transition probability and update the state."""
    current_wind_speed, current_demand = state[1], state[2]
    
    # Transition wind speed and demand
    new_wind_speed = wind_speed_transition(current_wind_speed, mu_w, sigma_w, alpha, t)
    new_demand = energy_demand_transition(current_demand, mu_d, sigma_d, alpha, t)
    
    # Update phi based on action
    new_phi = apply_action(state, action)
    
    # Return new state
    return (new_phi, new_wind_speed, new_demand)  

# REWARD FUNCTION, NEED TO FIX
def calculate_reward(state, action):
    phi, wind_speed, demand = state[0], state[1], state[2]
    power_output = calculate_power_output(wind_speed)   
   
    total_energy_supply = power_output + phi * demand
    reward = 0

    if total_energy_supply < demand:
        reward += -100
    else:
        reward += 25 * (1 - phi) - 5

    # reward stability
    if action == "no_change":
        reward += 1
    return reward

def descritize_state(state):
    phi, wind_speed, demand = state
    wind_bin = wind_speed//1
    #bins = np.linspace(0, 4, 1)
    bins = [-np.inf, 3.4, 3.8, 4.2, 4.6, np.inf]
    return (phi, wind_bin, np.digitize(demand, bins))   

# Using epsilon greedy policy exploration to balance exploration and exploitation so agent doesn't get stuck in sub-optimal action plans
def e_greedy_policy(Q, state, epsilon):
    state = descritize_state(state)  
    if random.uniform(0, 1) < epsilon:
        return random.choice(['increase', 'decrease', 'no_change']) 
    return max(Q[state], key=Q[state].get)   

def q_learning(df, Q, num_episodes, step_horizon, mu_d, sigma_d, alpha, gamma, epsilon):
    mu_w = df["wind_speed"].mean()  
    sigma_w = df["wind_speed"].std()    

    for episode in range(num_episodes):
        
        curr_row = df.sample().iloc[0]
        state = descritize_state(curr_row["state"])
        x = []
        y = []
        for step in range(step_horizon):
            action = e_greedy_policy(Q, state, epsilon)
            new_state = full_transition(state, action, mu_w = mu_w
                                        , sigma_w = sigma_w, mu_d = mu_d, 
                                        sigma_d = sigma_d, alpha=0.1, t=1)
            if new_state[0] == 1 and action == 'increase':
                action = 'no_change'  

            new_state = descritize_state(new_state)  
            reward = calculate_reward(new_state, action) 
            
            best_next_action = max(Q[new_state], key=Q[new_state].get)
            Q[state][action] += alpha * (reward + gamma * Q[new_state][best_next_action] - Q[state][action])

 
            # x.append(step)  
            # y.append(state[0])  
            state = new_state
            if step % 99 == 0:
                print(f"Episode: {episode}/{num_episodes}, Step: {step}/{step_horizon}, State: {state}, Action: {action}, Reward: {reward}")
        
        
        # plt.plot(x, y, color = 'blue', alpha = 0.05)
        
        if episode % 1000 == 0:
            print(f"Episode: {episode}/{num_episodes}, Step: {step}/{step_horizon}, State: {state}, Action: {action}, Reward: {reward}")


    # plt.ylabel("Proportion of Energy Met by Fossil Fuels")
    # plt.xticks([])
    # plt.xlabel("Step")
    # plt.title("Agent Exploration Over Time")
    # plt.show()
    return Q
   

def main():
    # Load data
    df = get_data()
    
    # capacity factor
    global CP 
    CP = df['capacity_factor'].mean()   

    # Print wind speed statistics
    wind_speed_stats = df["wind_speed"].describe()    
    print(wind_speed_stats)

    # create power output column
    df['power_output'] = df['wind_speed'].apply(calculate_power_output)
    



    # NEEDS TO BE UPDATED:
    # Luis/Jlee, ideally we start at 100% Fossil Fuels and work down until we reach the optimal balance
    df['phi'] = 0.5  # Example: 50% of energy demand initially met by fossil fuels
    mu_d = 3.87
    sigma_d = 0.2
    df['demand'] =  3.87

    # constructing state space
    df['state'] = df.apply(lambda row: (row['phi'], row['wind_speed'], row['demand']), axis=1)


    
    # action space (increase, decrease, no_change in fossil fuel proportion)
    actions = ['increase', 'decrease', 'no_change']

    """Q-lEARNING SECTION"""   
    # Initialize Q table with random posiitve rewards to encourage exploration
    Q = defaultdict(lambda: {action: random.uniform(0,1) for action in actions})

    # Parameters for Q-learning
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.3 # Exploration probability
    episodes = 100 # Number of episodes
    step_horizon = 1000  # Number of steps per episode

    
    Q = q_learning(df, Q, episodes, step_horizon=step_horizon, mu_d=mu_d, sigma_d=sigma_d, alpha=alpha, gamma=gamma, epsilon=epsilon) 
    optimal_policy = {state: max(actions, key = actions.get) for state, actions in Q.items()}   
    #print(optimal_policy)
    Q_heatmap(Q)  
    #phi_vs_ws_heatmap(optimal_policy)    
    #parallel_axes(optimal_policy)
    print(df)

if __name__ == "__main__":
    main()
