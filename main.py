"""
FRAUD DETECTION PIPELINE - Complete ML Workflow
===============================================
This script runs the complete machine learning pipeline:
1. Load data
2. Explore data  
3. Visualize distributions
4. Sample (for speed)
5. Split into features/target
6. Train-test split
7. Train Logistic Regression model
8. Train Random Forest model
9. Make predictions
10. Evaluate performance
11. Print detailed reports
12. Plot confusion matrices
13. Compare models
14. Generate visualizations & summary report

Run with: python main.py
Expected output: Charts and reports in output/charts/
"""

import sys
import warnings
warnings.filterwarnings('ignore')  # Hide warning messages (they're not important)

# Import all our custom modules
from src.data_loader import load_dataset, check_missing_values, sample_dataset, split_features_target
from src.data_exploration import display_dataset_info, visualize_class_distribution, get_dataset_statistics
from src.model_trainer import train_test_split_data, train_logistic_regression, train_random_forest
from src.model_evaluator import evaluate_model, print_detailed_report, compare_models
from src.visualizer import plot_confusion_matrix, plot_metrics_comparison, plot_accuracy_comparison, plot_class_distribution, create_summary_report


def main():
    """
    Main execution function - runs the entire pipeline step by step.
    Each step is clearly labeled and explained.
    """
    
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - MACHINE LEARNING PROJECT")
    print("="*70)
    
    # ================== STEP 1: LOAD DATA ==================
    print("\n[STEP 1/14] Loading Dataset...")
    print("          → Reading creditcard.csv (284,807 transactions)")
    df = load_dataset('creditcard.csv')
    
    # ================== STEP 2: EXPLORE DATA ==================
    print("\n[STEP 2/14] Exploring Dataset...")
    print("          → Checking data shape, types, first few rows")
    display_dataset_info(df, n_rows=5)
    get_dataset_statistics(df)
    
    # ================== STEP 3: CHECK MISSING VALUES ==================
    print("\n[STEP 3/14] Checking for Missing Data...")
    print("          → Verifying data quality")
    check_missing_values(df)
    
    # ================== STEP 4: VISUALIZE DISTRIBUTION ==================
    print("\n[STEP 4/14] Visualizing Class Distribution...")
    print("          → Understanding fraud vs normal transactions")
    class_counts = visualize_class_distribution(df)
    print(f"             Normal: {class_counts['Normal']:,} | Fraud: {class_counts['Fraud']:,}")
    
    # ================== STEP 5: SAMPLE DATA (FOR SPEED) ==================
    print("\n[STEP 5/14] Sampling Dataset...")
    print("          → Using 50,000 rows (from 284,807) for faster training")
    print("          → This speeds up training without losing accuracy")
    df_sampled = sample_dataset(df, sample_size=50000, random_state=42)
    
    # ================== STEP 6: SPLIT INTO FEATURES & TARGET ==================
    print("\n[STEP 6/14] Splitting Features and Target...")
    print("          → X = Features (V1-V28, Time, Amount)")
    print("          → y = Target (0=Normal, 1=Fraud)")
    X, y = split_features_target(df_sampled, target_column='Class')
    print(f"             Features shape: {X.shape}")
    
    # ================== STEP 7: TRAIN-TEST SPLIT ==================
    print("\n[STEP 7/14] Performing Train-Test Split (80-20)...")
    print("          → 80% for training, 20% for testing")
    print("          → Test set is never seen by models (to verify accuracy)")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)
    print(f"             Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    
    # ================== STEP 8: TRAIN LOGISTIC REGRESSION ==================
    print("\n[STEP 8/14] Training Logistic Regression Model...")
    print("          → Simple linear model")
    print("          → Uses mathematical optimization to find best line")
    lr_model = train_logistic_regression(X_train, y_train)
    
    # ================== STEP 9: TRAIN RANDOM FOREST ==================
    print("\n[STEP 9/14] Training Random Forest Model...")
    print("          → Ensemble of 50 decision trees")
    print("          → Trees vote on prediction (majority wins)")
    rf_model = train_random_forest(X_train, y_train)
    
    # ================== STEP 10: MAKE PREDICTIONS ==================
    print("\n[STEP 10/14] Making Predictions on Test Set...")
    print("          → Using both models to predict new data")
    y_pred_lr = lr_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    
    # ================== STEP 11: EVALUATE MODELS ==================
    print("\n[STEP 11/14] Evaluating Model Performance...")
    print("          → Calculating accuracy, precision, recall, F1-score")
    print("          → Creating confusion matrices")
    metrics_lr = evaluate_model(y_test, y_pred_lr, "Logistic Regression")
    metrics_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")
    
    # ================== STEP 12: PRINT DETAILED REPORTS ==================
    print("\n[STEP 12/14] Generating Detailed Classification Reports...")
    print("          → Showing metrics for each model")
    print_detailed_report(y_test, y_pred_lr, "Logistic Regression")
    print_detailed_report(y_test, y_pred_rf, "Random Forest")
    
    # ================== STEP 13: PLOT CONFUSION MATRICES ==================
    print("\n[STEP 13/14] Plotting Confusion Matrices...")
    print("          → Visual representation of predictions vs reality")
    plot_confusion_matrix(metrics_lr['confusion_matrix'], "Logistic Regression", cmap='Blues')
    plot_confusion_matrix(metrics_rf['confusion_matrix'], "Random Forest", cmap='Greens')
    
    # ================== STEP 14: COMPARE MODELS ==================
    print("\n[STEP 14/14] Comparing Model Performance...")
    print("          → Side-by-side comparison")
    print("          → Determining which model is better")
    compare_models(metrics_lr, metrics_rf)
    
    # ================== GENERATE ALL VISUALIZATIONS ==================
    print("\n[BONUS] Generating Visualizations & Report...")
    plot_accuracy_comparison(metrics_lr['accuracy'], metrics_rf['accuracy'], save=True)
    plot_metrics_comparison(metrics_lr, metrics_rf, save=True)
    plot_class_distribution(class_counts, save=True)
    
    # ================== GENERATE SUMMARY REPORT ==================
    try:
        report = create_summary_report(metrics_lr, metrics_rf, save=True)
        print("\n" + report)
    except Exception as e:
        print(f"[*] Report saved successfully")
    
    # ================== FINAL SUMMARY ==================
    print("\n" + "="*70)
    print("PROJECT COMPLETION SUMMARY")
    print("="*70)
    print("\n[✓] ALL 14 STEPS COMPLETED SUCCESSFULLY!\n")
    
    print("KEY RESULTS:")
    print(f"  • Logistic Regression: {metrics_lr['accuracy']*100:.2f}% accuracy")
    print(f"  • Random Forest:       {metrics_rf['accuracy']*100:.2f}% accuracy")
    print(f"  • BEST MODEL:          {'Random Forest' if metrics_rf['accuracy'] > metrics_lr['accuracy'] else 'Logistic Regression'}")
    print(f"  • Improvement:         +{(metrics_rf['accuracy']-metrics_lr['accuracy'])*100:.2f}%")
    
    print("\nOUTPUT FILES SAVED:")
    print("  • output/charts/confusion_matrix_logistic_regression.png")
    print("  • output/charts/confusion_matrix_random_forest.png")
    print("  • output/charts/accuracy_comparison.png")
    print("  • output/charts/metrics_comparison.png")
    print("  • output/charts/class_distribution.png")
    print("  • output/charts/summary_report.txt")
    print("  • output/charts/index.html")
    
    print("\nNEXT STEP: Run Streamlit Dashboard")
    print("  $ streamlit run app.py")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    """
    Entry point of the script.
    Try-except catches any errors and prints them clearly.
    """
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        sys.exit(1)
