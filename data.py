import pandas as pd
import requests
from io import StringIO 

def get_data():
    url = "https://data.nrel.gov/system/files/54/wtk_site_metadata.csv"
    response = requests.get(url)    
    content = response.content.decode('utf-8')  

    raw = pd.read_csv(StringIO(content))  
    df = raw[["site_id", "longitude", "latitude", "power_curve", "wind_speed", "capacity_factor", "full_timeseries_path"]] 
    return df   
