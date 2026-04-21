"""
Data exploration module for credit card fraud detection project.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def display_dataset_info(df: pd.DataFrame, n_rows: int = 5) -> None:
    """
    Display first few rows and basic dataset information.
    
    Args:
        df (pd.DataFrame): Input dataset
        n_rows (int): Number of rows to display
    """
    print("="*70)
    print("DATASET OVERVIEW")
    print("="*70)
    print(f"\nFirst {n_rows} rows:")
    print(df.head(n_rows))
    print(f"\nDataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nDataset info:")
    print(df.info())
    print("="*70)


def visualize_class_distribution(df: pd.DataFrame, target_column: str = 'Class', figsize: tuple = (10, 5), save: bool = False) -> dict:
    """
    Visualize fraud vs normal transactions distribution.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Name of target column
        figsize (tuple): Figure size
        save (bool): Whether to save the plot
        
    Returns:
        dict: Dictionary with class counts
    """
    plt.figure(figsize=figsize)
    sns.countplot(x=target_column, data=df, palette='Set1')
    plt.title('Fraud vs Normal Transactions Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class (0: Normal, 1: Fraud)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    print(f"\nCounts:")
    counts = df[target_column].value_counts()
    print(counts)
    print(f"\nPercentages:")
    print(df[target_column].value_counts(normalize=True) * 100)
    print("="*70)
    
    # Return class counts as dictionary
    class_dict = {
        'Normal': int(counts.get(0, 0)),
        'Fraud': int(counts.get(1, 0))
    }
    
    return class_dict


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Get descriptive statistics of the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dictionary with statistics
    """
    stats = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'numeric_cols': len(df.select_dtypes(include=['number']).columns)
    }
    
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Shape: {stats['shape'][0]:,} rows, {stats['shape'][1]} columns")
    print(f"Missing values: {stats['missing_values']}")
    print(f"Duplicate rows: {stats['duplicates']}")
    print(f"Numeric columns: {stats['numeric_cols']}")
    print("="*70)
    
    return stats
