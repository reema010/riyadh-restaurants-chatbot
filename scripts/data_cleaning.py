"""
Data Cleaning Script

This script was used to clean and preprocess the original Riyadh restaurants dataset
before exporting the cleaned CSV used by the chatbot.
"""

import pandas as pd

REQUIRED_COLUMNS = {"name", "categories", "address", "lat", "lng", "price", "rating"}

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.drop_duplicates()
    df = df.dropna(subset=["name", "lat", "lng"])

    for col in ["name", "categories", "address", "price"]:
        df[col] = df[col].astype(str).str.strip()

    df["name_l"] = df["name"].str.lower()
    df["categories_l"] = df["categories"].str.lower()
    df["address_l"] = df["address"].str.lower()
    df["price_l"] = df["price"].str.lower()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)

    df = df[(df["lat"].between(-90, 90)) & (df["lng"].between(-180, 180))]
    df = df.reset_index(drop=True)

    return df
