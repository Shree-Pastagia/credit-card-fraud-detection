"""
Model evaluation module for credit card fraud detection project.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Evaluate model performance and return metrics.
    
    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary containing all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    print(f"\n[OK] {model_name} - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return metrics


def print_detailed_report(y_true: pd.Series, y_pred: np.ndarray, model_name: str = "Model") -> None:
    """
    Print detailed classification report.
    
    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
        model_name (str): Name of the model
    """
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION REPORT - {model_name.upper()}")
    print(f"{'='*70}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    print(f"{'='*70}")


def print_confusion_matrix(cm: np.ndarray, model_name: str = "Model") -> None:
    """
    Print confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        model_name (str): Name of the model
    """
    print(f"\nConfusion Matrix ({model_name}):")
    print(cm)
    print(f"\nTrue Negatives: {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives: {cm[1,1]}")


def compare_models(metrics_lr: dict, metrics_rf: dict) -> None:
    """
    Compare two models' metrics.
    
    Args:
        metrics_lr (dict): Logistic Regression metrics
        metrics_rf (dict): Random Forest metrics
    """
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<20} {'Logistic Regression':<25} {'Random Forest':<25}")
    print("-"*70)
    print(f"{'Accuracy':<20} {metrics_lr['accuracy']:<25.4f} {metrics_rf['accuracy']:<25.4f}")
    print(f"{'Precision':<20} {metrics_lr['precision']:<25.4f} {metrics_rf['precision']:<25.4f}")
    print(f"{'Recall':<20} {metrics_lr['recall']:<25.4f} {metrics_rf['recall']:<25.4f}")
    print(f"{'F1-Score':<20} {metrics_lr['f1_score']:<25.4f} {metrics_rf['f1_score']:<25.4f}")
    print("="*70)
    
    # Determine better model
    if metrics_rf['accuracy'] > metrics_lr['accuracy']:
        diff = (metrics_rf['accuracy'] - metrics_lr['accuracy']) * 100
        print(f"\n[OK] Random Forest performs better by {diff:.2f}%")
    else:
        diff = (metrics_lr['accuracy'] - metrics_rf['accuracy']) * 100
        print(f"\n[OK] Logistic Regression performs better by {diff:.2f}%")
    print("="*70)
