from packages import *


def phi_vs_ws_heatmap(policy_matrix):    
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

    cmap = ListedColormap(['green', 'yellow', 'red']) 
  
    sns.heatmap(pm, xticklabels=wind_speeds, yticklabels=phis, cmap = cmap , cbar=False)   
    plt.title("Optimal Policy for Proportion of Energy Met by Fossil Fuels")
    plt.xlabel("Wind Speed")    
    plt.ylabel("Proportion of Energy Met by Fossil Fuels")  
    plt.legend(handles=[Patch(facecolor='green', label='Increase Renewables'), Patch(facecolor='yellow', label='No Change'), Patch(facecolor='red', label='Decrease Renewables')], loc='upper left')
    plt.show()

def Q_heatmap(Q):
    selected_action = "no_change"

    data =[]
    for (phi, ws, d) in Q.keys():
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
    
    sns.heatmap(Q_matrix, xticklabels=wind_speeds, yticklabels=phis, vmax = 20, vmin = -400, cmap = "coolwarm_r", cbar_kws={'label': 'Q-value'})   
    plt.xlabel("Wind Speed")    
    plt.ylabel("Proportion of Energy Met by Fossil Fuels")  
    plt.title(f"Q-values for fossil fuel action: [{selected_action}]") 
    plt.show()


def parallel_axes(policy_matrix):
    data = []
    for (phi, ws, d), action in policy_matrix.items():
        data.append({"phi": phi, "ws": ws, "d": d, "action": action})
    df = pd.DataFrame(data) 
    df["action"] = df["action"].astype("category")
    
    plt.figure(figsize=(10,6))

    parallel_coordinates(df, class_column='action', cols=['phi', 'ws', 'd'], color=('#1f77b4', '#ff7f0e', '#2ca02c'))

    plt.title("Parallel Coordinates of States by Action")
    plt.xlabel("State Dimensions")
    plt.ylabel("Values")
    plt.grid(True)
    plt.show()