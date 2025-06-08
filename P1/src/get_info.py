#!/usr/bin/env python
import sys
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_csv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    df = pd.read_csv(input_file)
    
    # Select categorical columns. This will include object, bool, and category types.
    categorical_columns = df.select_dtypes(include=["object", "bool", "category"]).columns
    
    # For each categorical column, output its unique values
    for col in categorical_columns:
        # Get unique values; sort them (as strings) for easier readability.
        # Convert to string to avoid comparison issues between booleans and strings.
        unique_vals = sorted(df[col].dropna().astype(str).unique())
        print(f"{col}: {unique_vals}")

if __name__ == "__main__":
    main()
