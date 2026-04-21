import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path('output/charts')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")

def plot_confusion_matrix(cm, model_name, figsize=(10, 8), cmap='Blues', save=True):
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'], cbar_kws={'label': 'Count'}, linewidths=2)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    if save:
        filepath = OUTPUT_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filepath}")
    plt.show()

def plot_metrics_comparison(metrics_lr, metrics_rf, figsize=(12, 6), save=True):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    lr_vals = [metrics_lr['accuracy'], metrics_lr['precision'], metrics_lr['recall'], metrics_lr['f1_score']]
    rf_vals = [metrics_rf['accuracy'], metrics_rf['precision'], metrics_rf['recall'], metrics_rf['f1_score']]
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, lr_vals, width, label='Logistic Regression', color='#3498db', alpha=0.85)
    ax.bar(x + width/2, rf_vals, width, label='Random Forest', color='#2ecc71', alpha=0.85)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0.98, 1.005])
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {OUTPUT_DIR / 'metrics_comparison.png'}")
    plt.show()

def plot_accuracy_comparison(accuracy_lr, accuracy_rf, figsize=(10, 6), save=True):
    models = ['Logistic Regression', 'Random Forest']
    accuracies = [accuracy_lr, accuracy_rf]
    colors = ['#3498db', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(models, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0.999, 1.001])
    for i, (m, a) in enumerate(zip(models, accuracies)):
        ax.text(i, a + 0.00001, f'{a:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {OUTPUT_DIR / 'accuracy_comparison.png'}")
    plt.show()

def plot_class_distribution(class_counts, figsize=(10, 6), save=True):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['#2ecc71', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(classes, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = (count / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {OUTPUT_DIR / 'class_distribution.png'}")
    plt.show()

def create_summary_report(metrics_lr, metrics_rf, save=True):
    from datetime import datetime
    timestamp = datetime.now().strftime('%B %d, %Y')
    
    lr_tn, lr_fp, lr_fn, lr_tp = metrics_lr['confusion_matrix'].flatten()
    rf_tn, rf_fp, rf_fn, rf_tp = metrics_rf['confusion_matrix'].flatten()
    
    report = f"""FRAUD DETECTION - SUMMARY REPORT
Generated: {timestamp}

RESULTS
Logistic Regression: {metrics_lr['accuracy']*100:.2f}% Accuracy
Random Forest: {metrics_rf['accuracy']*100:.2f}% Accuracy (BEST)

DATASET
Total: 284,807 | Fraud: 492 (0.17%) | Normal: 284,315 (99.83%)
Train: 40,000 | Test: 10,000

MODEL COMPARISON
                 LR           RF
Accuracy         {metrics_lr['accuracy']:.4f}      {metrics_rf['accuracy']:.4f}
Precision        {metrics_lr['precision']:.4f}      {metrics_rf['precision']:.4f}
Recall           {metrics_lr['recall']:.4f}      {metrics_rf['recall']:.4f}
F1-Score         {metrics_lr['f1_score']:.4f}      {metrics_rf['f1_score']:.4f}

CONFUSION MATRIX
Logistic Regression: TN={int(lr_tn)}, TP={int(lr_tp)}, FP={int(lr_fp)}, FN={int(lr_fn)}
Random Forest: TN={int(rf_tn)}, TP={int(rf_tp)}, FP={int(rf_fp)}, FN={int(rf_fn)}

CONCLUSION
Random Forest is recommended. Catches {int(rf_tp)} frauds vs {int(lr_tp)} (LR).
Misses only {int(rf_fn)} vs {int(lr_fn)} (LR).
"""
    
    if save:
        filepath = OUTPUT_DIR / "summary_report.txt"
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"[OK] Saved: {filepath}")
    
    return report

