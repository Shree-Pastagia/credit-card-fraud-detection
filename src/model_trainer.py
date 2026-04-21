"""
Model training module for credit card fraud detection project.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"[OK] Train-test split (80-20):")
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Testing set: {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, max_iter: int = 1000):
    """
    Train Logistic Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        max_iter (int): Maximum iterations
        
    Returns:
        LogisticRegression: Trained model
    """
    print("\n" + "="*70)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*70)
    
    lr_model = LogisticRegression(random_state=42, max_iter=max_iter)
    lr_model.fit(X_train, y_train)
    
    print("[OK] Logistic Regression model trained successfully!")
    print("="*70)
    
    return lr_model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 50):
    """
    Train Random Forest Classifier model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of trees
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST")
    print("="*70)
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    print("[OK] Random Forest model trained successfully!")
    print("="*70)
    
    return rf_model


def make_predictions(model, X_test: pd.DataFrame):
    """
    Make predictions on test data.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        
    Returns:
        np.ndarray: Predictions
    """
    predictions = model.predict(X_test)
    return predictions
