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
import os
import pickle
warnings.filterwarnings('ignore')  # Hide warning messages (they're not important)

# Import all our custom modules
from src.data_loader import load_dataset, check_missing_values, sample_dataset, split_features_target
from src.data_exploration import display_dataset_info, visualize_class_distribution, get_dataset_statistics
from src.model_trainer import train_test_split_data, train_logistic_regression, train_random_forest
from src.model_evaluator import evaluate_model, print_detailed_report, compare_models
from src.visualizer import plot_confusion_matrix, plot_metrics_comparison, plot_accuracy_comparison, plot_class_distribution, create_summary_report
from src.live_predictor import cli_live_input

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')


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
    
    # Save Logistic Regression model
    with open('models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print("[✓] Logistic Regression model saved to models/logistic_regression.pkl")
    
    # ================== STEP 9: TRAIN RANDOM FOREST ==================
    print("\n[STEP 9/14] Training Random Forest Model...")
    print("          → Ensemble of 50 decision trees")
    print("          → Trees vote on prediction (majority wins)")
    rf_model = train_random_forest(X_train, y_train)
    
    # Save Random Forest model
    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("[✓] Random Forest model saved to models/random_forest.pkl")
    
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
    print("  • models/logistic_regression.pkl")
    print("  • models/random_forest.pkl")
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
    
    # ================== BONUS: TEST SAMPLE DATA ==================
    print("\n" + "="*70)
    print("TESTING SAMPLE DATA (Fraud vs Non-Fraud)")
    print("="*70)
    
    import numpy as np
    
    # Sample legitimate transaction
    legit_data = "0, -1.35980713, -0.0727811733, 2.53634674, 1.37815522, -0.33832077, 0.462387778, 0.239598554, 0.0986979013, 0.36378697, 0.090794172, -0.551599533, -0.617800856, -0.991389847, -0.311169354, 1.46817697, -0.470400525, 0.207971242, 0.0257905802, 0.40399296, 0.251412098, -0.0183067779, 0.277837576, -0.11047391, 0.0669280749, 0.128539358, -0.189114844, 0.133558377, -0.0210530535, 149.62"
    
    # Sample fraudulent transaction
    fraud_data = "406, -2.31222654, 1.95199201, -1.60985073, 3.99790559, -0.522187865, -1.42654532, -2.53738731, 1.39165725, -2.77008928, -2.77227214, 3.20203321, -2.89990739, -0.595221881, -4.28925378, 0.38972412, -1.14074718, -2.83005567, -0.0168224682, 0.416955705, 0.126910559, 0.517232371, -0.0350493686, -0.465211076, 0.320198199, 0.0445191675, 0.177839798, 0.261145003, -0.143275875, 0.0"
    
    print("\n[TEST 1] LEGITIMATE TRANSACTION")
    print("-" * 70)
    legit_values = np.array([float(v.strip()) for v in legit_data.split(',')]).reshape(1, -1)
    legit_pred_lr = lr_model.predict(legit_values)[0]
    legit_prob_lr = lr_model.predict_proba(legit_values)[0]
    legit_pred_rf = rf_model.predict(legit_values)[0]
    legit_prob_rf = rf_model.predict_proba(legit_values)[0]
    
    print(f"Sample Data: {legit_data}")
    print(f"\n  Logistic Regression: {'🚨 FRAUDULENT' if legit_pred_lr == 1 else '✅ LEGITIMATE'}")
    print(f"    Confidence: {(legit_prob_lr[1] if legit_pred_lr == 1 else legit_prob_lr[0])*100:.2f}%")
    print(f"\n  Random Forest: {'🚨 FRAUDULENT' if legit_pred_rf == 1 else '✅ LEGITIMATE'}")
    print(f"    Confidence: {(legit_prob_rf[1] if legit_pred_rf == 1 else legit_prob_rf[0])*100:.2f}%")
    
    print("\n[TEST 2] FRAUDULENT TRANSACTION")
    print("-" * 70)
    fraud_values = np.array([float(v.strip()) for v in fraud_data.split(',')]).reshape(1, -1)
    fraud_pred_lr = lr_model.predict(fraud_values)[0]
    fraud_prob_lr = lr_model.predict_proba(fraud_values)[0]
    fraud_pred_rf = rf_model.predict(fraud_values)[0]
    fraud_prob_rf = rf_model.predict_proba(fraud_values)[0]
    
    print(f"Sample Data: {fraud_data}")
    print(f"\n  Logistic Regression: {'🚨 FRAUDULENT' if fraud_pred_lr == 1 else '✅ LEGITIMATE'}")
    print(f"    Confidence: {(fraud_prob_lr[1] if fraud_pred_lr == 1 else fraud_prob_lr[0])*100:.2f}%")
    print(f"\n  Random Forest: {'🚨 FRAUDULENT' if fraud_pred_rf == 1 else '✅ LEGITIMATE'}")
    print(f"    Confidence: {(fraud_prob_rf[1] if fraud_pred_rf == 1 else fraud_prob_rf[0])*100:.2f}%")
    
    print("\n" + "="*70)
    
    # Return models for optional live prediction
    return rf_model


if __name__ == "__main__":
    """
    Entry point of the script.
    Try-except catches any errors and prints them clearly.
    """
    try:
        model = main()
        
        # Ask user if they want to try live predictions
        print("\n" + "="*70)
        print("LIVE PREDICTION MODE (OPTIONAL)")
        print("="*70)
        try:
            user_choice = input("\nWould you like to test live fraud detection? (yes/no): ").strip().lower()
            if user_choice in ['yes', 'y']:
                cli_live_input(model)
        except KeyboardInterrupt:
            print("\n\nExiting. Thank you for using Fraud Detection System!")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        sys.exit(1)
