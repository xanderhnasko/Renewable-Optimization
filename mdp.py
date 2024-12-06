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
    return max(np.random.normal(mu_w_prime, sigma_w), 0)

# this is placeholder code for now 
# we also need something for mu_d, sigma_d
def energy_demand_transition(current_demand, mu_d, sigma_d, alpha=0.1, t=1):
    lambda_ = np.exp(-alpha * t)
    mu_d_prime = lambda_ * current_demand + (1 - lambda_) * mu_d
    return max(np.random.normal(mu_d_prime, sigma_d), 0)

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
    new_demand = round(energy_demand_transition(current_demand, mu_d, sigma_d, alpha, t), 3)
    
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

    bins = [0, 25, 50, 75, 100, 125, 150, 200]
    demand_bin = np.digitize(demand, bins)
    return (phi, wind_bin, demand_bin)   

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
            reward = calculate_reward(new_state, action) 
            new_state = descritize_state(new_state) 

            best_next_action = max(Q[new_state], key=Q[new_state].get)
            Q[state][action] += alpha * (reward + gamma * Q[new_state][best_next_action] - Q[state][action])

 
            x.append(step)  
            y.append(state[0])  
            state = new_state
            if step % 99 == 0:
                print(f"Episode: {episode}/{num_episodes}, Step: {step}/{step_horizon}, State: {state}, Action: {action}, Reward: {reward}")
        plt.plot(x, y, color = 'blue', alpha = 0.01)
        
        if episode % 1000 == 0:
            print(f"Episode: {episode}/{num_episodes}, Step: {step}/{step_horizon}, State: {state}, Action: {action}, Reward: {reward}")

    #sns.kdeplot(x = x_data, y = y_data, cmap = "Blues", fill = True)    
    plt.xlabel("Step")  
    plt.ylabel("Proportion of Energy Met by Fossil Fuels")
    plt.show()
    return Q

def visualize_optimal_ws_d(policy_matrix): 
    action_map = {"decrease": 0, "no_change": 1, "increase": 2} 
    
    data = [(d, ws, action_map[action]) for (phi, ws, d), action in policy_matrix.items()]  
    if not data:
        print("No data to visualize")
        return
    
    ds = sorted(list(set([row[0] for row in data])))  
    wind_speeds = sorted(list(set([row[1] for row in data])))   

    d_index = {d: i for i, d in enumerate(ds)}  
    wind_speed_index = {ws: i for i, ws in enumerate(wind_speeds)}  

    pm = np.zeros((len(ds), len(wind_speeds))) 

    for (phi, ws, action) in data:  
        i = d_index[phi]
        j = wind_speed_index[ws]
        pm[i, j] = action   

    cmap = ListedColormap(['green', 'white', 'red']) 
  
    sns.heatmap(pm, xticklabels=wind_speeds, yticklabels=ds, cmap = cmap , cbar=False) 
    plt.title("Optimal Policy for Energy Demand and Wind Speed")    
    plt.xlabel("Wind Speed")    
    plt.ylabel("Energy Demand") 
    plt.show()

def visualize_optimal_phi_ws(policy_matrix):    
    action_map = {"decrease": 0, "no_change": 1, "increase": 2} 
    
    data = [(phi, ws, action_map[action]) for (phi, ws, d), action in policy_matrix.items()]  
    if not data:
        print("No data to visualize")
        return
    
    phis = sorted(list(set([row[0] for row in data])))  
    wind_speeds = sorted(list(set([row[1] for row in data])))   

    phi_index = {phi: i for i, phi in enumerate(phis)}  
    wind_speed_index = {ws: i for i, ws in enumerate(wind_speeds)}  

    pm = np.zeros((len(phis), len(wind_speeds))) 

    for (phi, ws, action) in data:  
        i = phi_index[phi]
        j = wind_speed_index[ws]
        pm[i, j] = action   

    cmap = ListedColormap(['green', 'white', 'red']) 
  
    sns.heatmap(pm, xticklabels=wind_speeds, yticklabels=phis, cmap = cmap , cbar=False)   
    plt.title("Optimal Policy for Proportion of Energy Met by Fossil Fuels")
    plt.xlabel("Wind Speed")    
    plt.ylabel("Proportion of Energy Met by Fossil Fuels")  
    plt.show()

def visualize_Q(Q):
    selected_action = "no_change"

    data =[]
    for (phi, ws, d) in Q.keys():
        #if d == 2:   
        value = Q[(phi, ws, d)][selected_action] 
        data.append((phi, ws, value))

    phis = sorted(list(set([row[0] for row in data]))) 
    wind_speeds = sorted(list(set([row[1] for row in data])))

    phi_index = {phi: i for i, phi in enumerate(phis)}  
    wind_speed_index = {ws: i for i, ws in enumerate(wind_speeds)}  

    Q_matrix = np.zeros((len(phis), len(wind_speeds))) 

    for (phi, ws, q_val) in data:
        i = phi_index[phi]
        j = wind_speed_index[ws]
        Q_matrix[i, j] = q_val
    
    sns.heatmap(Q_matrix, xticklabels=wind_speeds, yticklabels=phis, vmin = -400, vmax = 20)   
    plt.xlabel("Wind Speed")    
    plt.ylabel("Proportion of Energy Met by Fossil Fuels")  
    plt.title(f"Q-values for [{selected_action}] action") 
    plt.show()

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
    mu_d = 100
    sigma_d = 10
    df['demand'] =  100 

    # constructing state space
    df['state'] = df.apply(lambda row: (row['phi'], row['wind_speed'], row['demand']), axis=1)
    print(df)
    
    # action space (increase, decrease, no_change in fossil fuel proportion)
    actions = ['increase', 'decrease', 'no_change']

    """Q-lEARNING SECTION"""   
    # Initialize Q table with random posiitve rewards to encourage exploration
    Q = defaultdict(lambda: {action: random.uniform(0,1) for action in actions})

    # Parameters for Q-learning
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.3 # Exploration probability
    episodes = 1000  # Number of episodes
    step_horizon = 100  # Number of steps per episode

    
    Q = q_learning(df, Q, episodes, step_horizon=step_horizon, mu_d=mu_d, sigma_d=sigma_d, alpha=alpha, gamma=gamma, epsilon=epsilon) 
    optimal_policy = {state: max(actions, key = actions.get) for state, actions in Q.items()}   
    print(optimal_policy) 
    visualize_Q(Q)  
    visualize_optimal_phi_ws(optimal_policy)    
    #visualize_optimal_ws_d(optimal_policy)

if __name__ == "__main__":
    main()
