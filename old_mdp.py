from packages import *
from data import get_data

def get_data(): 
    if not os.path.exists("data.pkl"):
        get_data()  
    df = pd.read_pickle("data.pkl") 
    column = df["wind_speed"]   
    print(column.describe())
    df["prefix"] = df["full_timeseries_path"].str.split("/", expand=True)[0]  
    print(df)  

    # JLEE - This just groups all sites by 1x1 degree squares, and outputs teh wind speed at each
    # But in this case, we might have to have some notion of location in our MDP
    # or just run the MDP on each location 
    df["lat_bin"] = df["latitude"]//1
    df["long_bin"] = df["longitude"]//1
    df["region_id"] = df["lat_bin"].astype(str) + "/" + df["long_bin"].astype(str)
    grouped = df.groupby("region_id")
    for region, group in grouped:
        print(f"region: {region}")
        print(group)
    


    # JLEE - this is the average wind speed (and variance) across all locations in the US. We can use this alone if need be,
    # but it doesn't make much practical sense to compare wind speeds that are far away from eachother. Would be really easy to make the MDP tho
    plt.hist(column, bins=20, color='c', edgecolor='black')   
    plt.title(f'Histogram of Wind Speed, mu={column.mean():.2f}, sigma={column.std():.2f}')    
    plt.xlabel('Wind Speed (m/s)')      
    plt.ylabel('Frequency') 
    plt.show()

def main():
    # Load data
    df = get_data()
    print(df.head())  # Print the first few rows to verify data loading 

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

    # Plot histogram of wind speed using overall dataset statistics
    plt.hist(df["wind_speed"], bins=20, color='c', edgecolor='black')
    plt.title(f'Histogram of Wind Speed, mu={wind_speed_stats["mean"]:.2f}, sigma={wind_speed_stats["std"]:.2f}')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')
    plt.show() 

if __name__ == "__main__":  
    main()
