from packages import *
from data import get_data   

# def get_data(): 
#     if not os.path.exists("data.pkl"):
#         get_data()  
#     df = pd.read_pickle("data.pkl") 
#     column = df["wind_speed"]   
#     print(column.describe())
#     df["prefix"] = df["full_timeseries_path"].str.split("/", expand=True)[0]  
#     print(df)  

#     # JLEE - This just groups all sites by 1x1 degree squares, and outputs teh wind speed at each
#     # But in this case, we might have to have some notion of location in our MDP
#     # or just run the MDP on each location 
#     df["lat_bin"] = df["latitude"]//1
#     df["long_bin"] = df["longitude"]//1
#     df["region_id"] = df["lat_bin"].astype(str) + "/" + df["long_bin"].astype(str)
#     grouped = df.groupby("region_id")
#     for region, group in grouped:
#         print(f"region: {region}")
#         print(group)
    


#     # JLEE - this is the average wind speed (and variance) across all locations in the US. We can use this alone if need be,
#     # but it doesn't make much practical sense to compare wind speeds that are far away from eachother. Would be really easy to make the MDP tho
#     plt.hist(column, bins=20, color='c', edgecolor='black')   
#     plt.title(f'Histogram of Wind Speed, mu={column.mean():.2f}, sigma={column.std():.2f}')    
#     plt.xlabel('Wind Speed (m/s)')      
#     plt.ylabel('Frequency') 
#     plt.show()

# def main():
#     get_data()  

# if __name__ == "__main__":  
#     main()

def get_data(filepath="data.pkl"):
    """Load the dataset from a pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError("Data file not found. Ensure the data has been downloaded and saved correctly.")
    return pd.read_pickle(filepath)

# Constants for the power output calculation
def calculate_power_output(wind_speed, Cp=0.5, rho=1.225, R=40):
    # Cp: i just made this up 
    # rho: from paper
    # R: made up
    return 0.5 * Cp * rho * np.pi * R**2 * wind_speed**3

# Xander- these mu and sigmas are from the average wind speed across all locations in the US 
# based on the histogram above
def wind_speed_transition(current_wind_speed, mu_w=7.58, sigma_w=1.02, alpha=0.1, t=1):
    lambda_ = 0.5 # Per luis' comments
    # NGl, the below line was made by Cursor. if we just use these made up alpha and t, it could work? 
    #lambda_ = np.exp(-alpha * t) 
    mu_w_prime = lambda_ * current_wind_speed + (1 - lambda_) * mu_w
    return np.random.normal(mu_w_prime, sigma_w)

# not sure what we are doing for demand, this is placeholder code for now 
# we also need something for mu_d, sigma_d
def energy_demand_transition(current_demand, mu_d, sigma_d, alpha=0.1, t=1):
    #lambda_ = np.exp(-alpha * t)
    lambda_ = 0.5
    mu_d_prime = lambda_ * current_demand + (1 - lambda_) * mu_d
    return np.random.normal(mu_d_prime, sigma_d)

# Constants for initial wind speed simulation
# Xander, these mu and sigmas are from the average wind speed across all locations in the US 
# based on the histogram the code is generating 
def simulate_initial_wind_speed(n, mu_w=7.58, sigma_w=1.02):
    # mu_w and sigma_w derived from overall dataset statistics
    return np.random.normal(mu_w, sigma_w, n)

# ACTION FUNCTIONS

def apply_action(row, action, delta=0.1):
    
    phi = row['phi']
    if action == 'increase':
        phi = min(phi + delta, 1)
    elif action == 'decrease':
        phi = max(phi - delta, 0)
    # 'no_change' action does nothing
    return phi

def full_transition(state, action, mu_w, sigma_w, mu_d, sigma_d, alpha=0.1, t=1):
    """Compute the full transition probability and update the state."""
    current_wind_speed, current_demand = state['wind_speed'], state['demand']
    
    # Transition wind speed and demand
    new_wind_speed = wind_speed_transition(current_wind_speed, mu_w, sigma_w, alpha, t)
    new_demand = energy_demand_transition(current_demand, mu_d, sigma_d, alpha, t)
    
    # Update phi based on action
    new_phi = apply_action(state, action)
    
    # Calculate new power output
    new_power_output = calculate_power_output(new_wind_speed)
    
    # Return new state
    return {
        'phi': new_phi,
        'wind_speed': new_wind_speed,
        'power_output': new_power_output,
        'demand': new_demand
    }

# REWARD FUNCTION
def calculate_reward(state, new_state):
    phi, wind_speed, power_output, demand = state['phi'], state['wind_speed'], state['power_output'], state['demand']
    new_phi, new_wind_speed, new_power_output, new_demand = new_state['phi'], new_state['wind_speed'], new_state['power_output'], new_state['demand']
    
    delta_phi = new_phi - phi
    total_energy_supply = new_power_output + new_phi * new_demand
    
    if total_energy_supply >= new_demand:
        return 10 * (1 - delta_phi)
    else:
        return -50 + 10 * (-delta_phi)

def main():
    # Load data
    df = get_data()

    # Print wind speed statistics
    wind_speed_stats = df["wind_speed"].describe()
    print(wind_speed_stats)

    # Add prefix from full_timeseries_path
    df["prefix"] = df["full_timeseries_path"].str.split("/", expand=True)[0]
    print(df.head())  # Print the first few rows to verify the prefix addition

    # Group by 1x1 degree squares and print grouped data
    df["lat_bin"] = df["latitude"] // 1
    df["long_bin"] = df["longitude"] // 1
    df["region_id"] = df["lat_bin"].astype(str) + "/" + df["long_bin"].astype(str)
    grouped = df.groupby("region_id")

    for region, group in grouped:
        print(f"Region: {region}")
        print(group)

    # # Plot histogram of wind speed using overall dataset statistics
    # plt.hist(df["wind_speed"], bins=20, color='c', edgecolor='black')
    # plt.title(f'Histogram of Wind Speed, mu={wind_speed_stats["mean"]:.2f}, sigma={wind_speed_stats["std"]:.2f}')
    # plt.xlabel('Wind Speed (m/s)')
    # plt.ylabel('Frequency')
    # plt.show()
    
     # power output shit from the paper 
    df['power_output'] = df['wind_speed'].apply(calculate_power_output)

    # NEEDS TO BE UPDATED:
    # i assumed constants here bc i was not entirelty sure how we calculate this
    df['phi'] = 0.7  # Example: 70% of energy demand met by fossil fuels
    df['demand'] = 100  # Example: 100 MW demand at each site

    # constructing state space
    df['state'] = df.apply(lambda row: (row['phi'], row['wind_speed'], row['power_output'], row['demand']), axis=1)

    # you can remove this if you don't want to see
    print(df[['state']].head())

    actions = ['increase', 'decrease', 'no_change']

    # this section is supposed to apply actions and calc. rewards
    # again, phi and demand are placeholders right now
    for action in actions:
        df['new_phi'] = df.apply(lambda row: apply_action(row, action), axis=1)
        df['new_wind_speed'] = df['wind_speed'].apply(lambda w: wind_speed_transition(w, mu_w=7.58, sigma_w=1.02))
        df['new_power_output'] = df['new_wind_speed'].apply(calculate_power_output)
        df['reward'] = df.apply(lambda row: calculate_reward(
            {'phi': row['phi'], 'wind_speed': row['wind_speed'], 'power_output': row['power_output'], 'demand': row['demand']},
            {'phi': row['new_phi'], 'wind_speed': row['new_wind_speed'], 'power_output': row['new_power_output'], 'demand': row['demand']}
        ), axis=1)
        print(f"Action: {action}, Rewards:\n{df['reward'].head()}")

if __name__ == "__main__":
    main()
