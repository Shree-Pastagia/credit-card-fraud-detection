# Credit Card Fraud Detection - Machine Learning Project

A comprehensive machine learning project for detecting fraudulent credit card transactions using Logistic Regression and Random Forest classifiers.

## Features

- **Machine Learning Models**: Logistic Regression & Random Forest
- **Interactive Web UI**: Streamlit dashboard with visualizations
- **Model Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Fraud Predictions**: Real-time fraud detection on new transactions
- **Data Exploration**: Interactive dataset analysis
- **Professional Reports**: Comprehensive summary reports and charts
- **Modular Code**: Organized, reusable Python modules

```
Fraud-Detection-Project/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── data_exploration.py      # Data exploration and statistics
│   ├── model_trainer.py         # Model training functions
│   ├── model_evaluator.py       # Model evaluation metrics
│   └── visualizer.py            # Visualization functions
├── data/
│   └── creditcard.csv           # Dataset (284,807 transactions)
├── main.py                      # Main execution script
├── fraud_detection.ipynb        # Jupyter notebook (reference)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Features

- **Data Loading & Exploration**: Load and analyze credit card transaction data
- **Data Preprocessing**: Handle missing values, sampling, feature/target split
- **Model Training**: Train both Logistic Regression and Random Forest models
- **Model Evaluation**: Calculate accuracy, precision, recall, and F1-score
- **Visualization**: Generate confusion matrices and performance comparison charts
- **Modular Design**: Organized code structure with separate modules for different tasks

## Dataset

- **File**: `creditcard.csv`
- **Size**: 284,807 transactions
- **Columns**: 31 (Time, V1-V28, Amount, Class)
- **Target**: Class (0: Normal, 1: Fraud)
- **Imbalance**: ~99.8% normal, ~0.2% fraud

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## Installation

1. Clone or download the project
2. Create a virtual environment (optional):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run the Streamlit Dashboard (Recommended)

**Windows:**
```bash
python -m streamlit run app.py
# Or double-click run_ui.bat
```

**macOS/Linux:**
```bash
streamlit run app.py
# Or bash run_ui.sh
```

The dashboard will open at `http://localhost:8501` with the following features:
- 📊 **Dashboard**: Overview of dataset and models
- 📈 **Model Performance**: Detailed metrics and confusion matrices
- 🔮 **Make Predictions**: Test fraud detection on new transactions
- 📁 **Dataset Explorer**: Analyze features and transactions
- 📄 **Summary Report**: View and download the comprehensive report

### Option 2: Run the Main Script

```bash
python main.py
```

This will execute all steps:
1. Load dataset
2. Explore data
3. Visualize class distribution
4. Sample dataset (50,000 rows)
5. Split features and target
6. Train-test split (80-20)
7. Train Logistic Regression
8. Train Random Forest
9. Make predictions
10. Evaluate models
11. Generate reports and visualizations

### Use the Jupyter Notebook

```bash
jupyter notebook fraud_detection.ipynb
```

### Import Modules in Your Code

```python
from src.data_loader import load_dataset, sample_dataset
from src.model_trainer import train_logistic_regression, train_random_forest
from src.model_evaluator import evaluate_model
```

## Results

### Model Performance

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 99.95% | 99.97% |
| Precision | 0.9994 | 0.9997 |
| Recall | 0.9995 | 0.9997 |
| F1-Score | 0.9994 | 0.9997 |

### Key Findings

- ✓ Random Forest slightly outperforms Logistic Regression
- ✓ Both models achieve excellent accuracy on this dataset
- ✓ Sampling strategy (50,000 rows) reduces training time while maintaining performance
- ✓ Random Forest detects more fraud cases with fewer false negatives

## Module Documentation

### data_loader.py
- `load_dataset()`: Load CSV dataset
- `check_missing_values()`: Check for missing values
- `sample_dataset()`: Randomly sample rows
- `split_features_target()`: Split into X and y

### data_exploration.py
- `display_dataset_info()`: Show first rows and info
- `visualize_class_distribution()`: Plot fraud vs normal
- `get_dataset_statistics()`: Get descriptive stats

### model_trainer.py
- `train_test_split_data()`: Split into train/test sets
- `train_logistic_regression()`: Train LR model
- `train_random_forest()`: Train RF model
- `make_predictions()`: Generate predictions

### model_evaluator.py
- `evaluate_model()`: Calculate metrics
- `print_detailed_report()`: Print classification report
- `print_confusion_matrix()`: Print confusion matrix
- `compare_models()`: Compare two models

### visualizer.py
- `plot_confusion_matrix()`: Heatmap visualization
- `plot_metrics_comparison()`: Compare metrics
- `plot_accuracy_comparison()`: Compare accuracies

## Future Improvements

- [ ] Add more classifiers (SVM, Gradient Boosting, Neural Networks)
- [ ] Implement feature scaling and normalization
- [ ] Add hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- [ ] Implement cross-validation
- [ ] Add ROC-AUC curves and other metrics
- [ ] Handle class imbalance (SMOTE, class weights)
- [ ] Add model persistence (save/load trained models)
- [ ] Create web API for predictions

## Author

Created as a comprehensive ML project for fraud detection

## License

Open source - feel free to use and modify

## Notes

- The dataset shows significant class imbalance (99.8% vs 0.2%)
- Current models handle this well but could benefit from balancing techniques
- All features (V1-V28) are pre-PCA transformed for privacy
- Sampling 50,000 rows maintains model accuracy while improving speed
