import pandas as pd
from sklearn.metrics import classification_report
import json

# Function to process a single fold
def process_fold(fold_number):
    file_path = f'/Users/mitul/Documents/study/sem 4/DSSE/Assignmets/Assignments 3/assignment3_repo/ds4se3-group6/utils/results/test_predictions_fold_{fold_number}.xlsx'
    data = pd.read_excel(file_path)
    true_labels = data['true_label']
    predicted_labels = data['predicted_label']
    report = classification_report(true_labels, predicted_labels, zero_division=0)
    return report

# Main script to process all folds and store results in a JSON file
def process_all_folds():
    results = {}
    for fold in range(1, 11):  # Assuming folds are numbered from 1 to 10
        fold_results = process_fold(fold)
        results[f'Fold_{fold}'] = fold_results
    
    # Save results to a JSON file
    with open('metrics_results.json', 'w') as fp:
        json.dump(results, fp, indent=4)

# Run the script
process_all_folds()
