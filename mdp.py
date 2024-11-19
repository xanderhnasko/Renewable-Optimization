from packages import *
from data import get_data   

def analyze_data(): 
    if not os.path.exists("data.pkl"):
        get_data()  
    df = pd.read_pickle("data.pkl") 
    column = df["wind_speed"]   
    print(column.describe())
    print(column.head())

    plt.hist(column, bins=20, color='c', edgecolor='black')   
    plt.title(f'Histogram of Wind Speed, mu={column.mean():.2f}, sigma={column.std():.2f}')    
    plt.xlabel('Wind Speed (m/s)')      
    plt.ylabel('Frequency') 
    plt.show()


def main():
    analyze_data()  

if __name__ == "__main__":  
    main()