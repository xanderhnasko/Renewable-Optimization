from packages import *
from data import get_data   

def analyze_data(): 
    if not os.path.exists("data.pkl"):
        get_data()  
    df = pd.read_pickle("data.pkl") 
    column = df["wind_speed"]   
    print(column.describe())
    print(column.head())


def main():
    analyze_data()  

if __name__ == "__main__":  
    main()