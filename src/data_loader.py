"""
Data loading module for credit card fraud detection project.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the credit card dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Dataset loaded successfully from: {filepath}")
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found at {filepath}")
        raise
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        raise


def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Check for missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dictionary with missing value statistics
    """
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    stats = {
        'total_missing': total_missing,
        'columns_with_missing': (missing_values > 0).sum(),
        'missing_details': missing_values[missing_values > 0]
    }
    
    print(f"[OK] Missing values check:")
    print(f"  Total missing values: {total_missing}")
    print(f"  Columns with missing values: {stats['columns_with_missing']}")
    
    return stats


def sample_dataset(df: pd.DataFrame, sample_size: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """
    Randomly sample rows from the dataset for faster training.
    
    Args:
        df (pd.DataFrame): Input dataset
        sample_size (int): Number of rows to sample
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Sampled dataset
    """
    df_sampled = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    reduction_pct = 100 * (1 - len(df_sampled) / len(df))
    
    print(f"[OK] Dataset sampling:")
    print(f"  Original size: {len(df):,} rows")
    print(f"  Sampled size: {len(df_sampled):,} rows")
    print(f"  Reduction: {reduction_pct:.2f}%")
    
    return df_sampled


def split_features_target(df: pd.DataFrame, target_column: str = 'Class'):
    """
    Split dataset into features and target variable.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Name of target column
        
    Returns:
        tuple: (features DataFrame, target Series)
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f"[OK] Features and target split:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    return X, y
