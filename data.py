
'''
The get_data function is used to download the data from the NREL website and save it as a pickle file.  
'''
from packages import *

def get_data():
    url = "https://data.nrel.gov/system/files/54/wtk_site_metadata.csv"
    response = requests.get(url)    
    content = response.content.decode('utf-8')  

    raw = pd.read_csv(StringIO(content))  
    df = raw[["site_id", "longitude", "latitude", "power_curve", "wind_speed", "capacity_factor", "full_timeseries_path"]] 
    df.to_pickle("data.pkl")    
