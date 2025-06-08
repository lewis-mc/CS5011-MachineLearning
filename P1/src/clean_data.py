import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("../data/train_values.csv")  

# List the columns you want to remove
columns_to_drop = ['date_recorded', 'wpt_name', 'recorded_by']

# Drop the specified columns from the DataFrame
df = df.drop(columns=columns_to_drop)

# List the columns to make into string
columns_to_string = ['region_code', 'district_code', 'public_meeting', 'permit', 'construction_year']

df[columns_to_string] = df[columns_to_string].astype(str)

# Optionally, save the modified DataFrame to a new CSV file
df.to_csv("../data/train_values_modified.csv", index=False)

print("Modified data frame created.")