# data_cleaning.py
import pandas as pd
import numpy as np

REQUIRES_COLUMNS = [
    "N", "P", "K",
    "temperature", "humidity", "ph", "rainfall",
    "label"
]

def clean_dataset(csv_path):
    # Cleans Crop dataset and returns a sanitised DataFrame.
    
    df = pd.read_csv(csv_path)
    
    # Column Validation
    missing_cols = set(REQUIRES_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing Columns: {missing_cols}")
    
    df = df[REQUIRES_COLUMNS]

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    # Type casting
    numeric_cols = REQUIRES_COLUMNS[:-1]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df =  df.dropna()

    # Domain-based filtering (IMPORTANT in AGRICULTURE)
    df = df[
        (df["N"] >= 0) & (df["N"] <= 200) &
        (df["P"] >= 0) & (df["P"] <= 200) &
        (df["K"] >= 0) & (df["K"] <= 200) &
        (df["temperature"] >= 0) & (df["temperature"] <= 60) &
        (df["humidity"] >= 0) & (df["humidity"] <= 100) &
        (df["ph"] >= 3) & (df["ph"] <= 10) &
        (df["rainfall"] >= 0) & (df["rainfall"] <= 5000)
    ]

    # Normalize crop labels
    df["label"] = df["label"].str.strip().str.lower()

    return df.reset_index(drop=True)