import pandas as pd
import os

def load_data(file) :

  if not os.path.exists(file) :
    raise FileNotFoundError(f"File {file} not found...")
  
  data = pd.read_excel(file)

  missing_count = data.isnull().sum().sum()

  if missing_count > 0 :
    print(f"[WARNING] missing data found. Delete rows with empty data....")
    
    data = data.dropna()

  print(f"[INFO] data is successfully loaded. The number of {file} data is {len(file)} rows")
  return data