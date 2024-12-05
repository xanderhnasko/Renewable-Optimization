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

    df["lat_bin"] = df["latitude"]//1
    df["long_bin"] = df["longitude"]//1
    df["region_id"] = df["lat_bin"].astype(str) + "/" + df["long_bin"].astype(str)
    grouped = df.groupby("region_id")
    for region, group in grouped:
        print(f"region: {region}")
        print(group)
    

    # plt.hist(column, bins=20, color='c', edgecolor='black')   
    # plt.title(f'Histogram of Wind Speed, mu={column.mean():.2f}, sigma={column.std():.2f}')    
    # plt.xlabel('Wind Speed (m/s)')      
    # plt.ylabel('Frequency') 
    # plt.show()
                
  

def main():
    get_data()  

if __name__ == "__main__":  
    main()