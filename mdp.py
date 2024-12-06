from packages import *
from data import get_data   
CP = 0.5

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
    return np.random.normal(mu_w_prime, sigma_w)

# this is placeholder code for now 
# we also need something for mu_d, sigma_d
def energy_demand_transition(current_demand, mu_d, sigma_d, alpha=0.1, t=1):
    lambda_ = np.exp(-alpha * t)
    mu_d_prime = lambda_ * current_demand + (1 - lambda_) * mu_d
    return np.random.normal(mu_d_prime, sigma_d)

# TBU Fix this, JLEE - where do we use this? is it necessary?
def simulate_initial_wind_speed(n, mu_w=7.58, sigma_w=1.02):
    # mu_w and sigma_w derived from overall dataset statistics
    return np.random.normal(mu_w, sigma_w, n)
                  
# ACTION FUNCTIONS

def apply_action(row, action, delta=0.01):
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
    current_wind_speed, current_demand = state[1], state[3]
    
    # Transition wind speed and demand
    new_wind_speed = wind_speed_transition(current_wind_speed, mu_w, sigma_w, alpha, t)
    new_demand = round(energy_demand_transition(current_demand, mu_d, sigma_d, alpha, t), 3)
    
    # Update phi based on action
    new_phi = apply_action(state, action)
    
    # Calculate new power output
    new_power_output = calculate_power_output(new_wind_speed)
    
    # Return new state
    return (new_phi, new_wind_speed, new_power_output, new_demand)  

# REWARD FUNCTION, NEED TO FIX
def calculate_reward(state, new_state):
    phi, wind_speed, power_output, demand = state[0], state[1], state[2], state[3]
    new_phi, new_wind_speed, new_power_output, new_demand = new_state[0], new_state[1], new_state[2], new_state[3]
    
    delta_phi = new_phi - phi
    total_energy_supply = new_power_output + new_phi * new_demand
    
    if total_energy_supply >= new_demand:
        return 10 * (1 - delta_phi)
    else:
        return -50 + 10 * (-delta_phi)

# Using epsilon greedy policy exploration to balance exploration and exploitation so agent doesn't get stuck in sub-optimal action plans
def e_greedy_policy(Q, state, epsilon): 
    if random.uniform(0, 1) < epsilon:
        return random.choice(['increase', 'decrease', 'no_change']) 
    return max(Q[state], key=Q[state].get)   

def q_learning(df, Q, num_episodes, step_horizon, mu_d, sigma_d, alpha, gamma, epsilon):
    mu_w = df["wind_speed"].mean()  
    sigma_w = df["wind_speed"].std()    

    for episode in range(num_episodes):
        
        curr_row = df.sample().iloc[0]
        state = curr_row["state"]
        x = []
        y = []
        for step in range(step_horizon):
            action = e_greedy_policy(Q, state, epsilon)
            new_state = full_transition(state, action, mu_w = mu_w
                                        , sigma_w = sigma_w, mu_d = mu_d, 
                                        sigma_d = sigma_d, alpha=0.1, t=1)
            reward = calculate_reward(state, new_state) 

            best_next_action = max(Q[new_state], key=Q[new_state].get)
            Q[state][action] += alpha * (reward + gamma * Q[new_state][best_next_action] - Q[state][action])

            x.append(step)
            y.append(state[0])
            state = new_state
            if step % 10 == 0:
                print(f"Episode: {episode}/{num_episodes}, Step: {step}/{step_horizon}, State: {state}, Action: {action}, Reward: {reward}")
        plt.plot(x, y)
        if episode % 100 == 0:
            print(f"Episode: {episode}/{num_episodes}, Step: {step}/{step_horizon}, State: {state}, Action: {action}, Reward: {reward}")

    plt.xlabel("Step")  
    plt.ylabel("Proportion of Energy Met by Fossil Fuels")
    plt.show()
    return Q

def visualize_policy(optimal_policy):
   pass
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
    df['phi'] = 0.7  # Example: 70% of energy demand initially met by fossil fuels
    mu_d = 100
    sigma_d = 10
    df['demand'] =  100 

    # constructing state space
    df['state'] = df.apply(lambda row: (row['phi'], row['wind_speed'], row['power_output'], row['demand']), axis=1)
    print(df)
    
    # action space (increase, decrease, no_change in fossil fuel proportion)
    actions = ['increase', 'decrease', 'no_change']

    """Q-lEARNING SECTION"""   
    # Initialize Q table
    Q = defaultdict(lambda: {action: 0.0 for action in actions})

    # Parameters for Q-learning
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration probability
    episodes = 500  # Number of episodes
    step_horizon = 100  # Number of steps per episode

    
    Q = q_learning(df, Q, episodes, step_horizon=step_horizon, mu_d=mu_d, sigma_d=sigma_d, alpha=alpha, gamma=gamma, epsilon=epsilon) 
    optimal_policy = {state: max(actions, key = actions.get) for state, actions in Q.items()}   
    print(optimal_policy)       


if __name__ == "__main__":
    main()
