import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load JSON data from file
with open('C:/DSSE/Assignment3/repo/ds4se3-group6/datasets/each_fold_results.json', 'r') as f:
    data = json.load(f)

# Initialize lists to store metrics
fold_data = []
epoch_data = []

# Extract fold-wise data
for fold_result in data['cross_validation_results']:
    fold = fold_result['fold']
    test_accuracy = fold_result['test_report']['test_accuracy']
    test_precision = fold_result['test_report']['test_precision']
    test_recall = fold_result['test_report']['test_recall']
    test_f1 = fold_result['test_report']['test_f1']
    
    fold_data.append({
        'Fold': fold,
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1-score': test_f1
    })
    
    # Extract epoch-wise validation data
    for epoch_result in fold_result['fold_results']:
        epoch = epoch_result['epoch']
        val_accuracy = epoch_result['val_accuracy']
        val_precision = epoch_result['val_precision']
        val_recall = epoch_result['val_recall']
        val_f1 = epoch_result['val_f1']
        
        epoch_data.append({
            'Fold': fold,
            'Epoch': epoch,
            'Validation Accuracy': val_accuracy,
            'Validation Precision': val_precision,
            'Validation Recall': val_recall,
            'Validation F1-score': val_f1
        })

# Convert to pandas DataFrame for easier manipulation
fold_df = pd.DataFrame(fold_data)
epoch_df = pd.DataFrame(epoch_data)

# Display metrics by fold and epoch
print("Metrics by Fold:")
print(fold_df)

print("\nMetrics by Epoch:")
print(epoch_df)

# Example: Plotting F1-score by epoch for each fold
plt.figure(figsize=(12, 6))
for fold in np.unique(epoch_df['Fold']):
    fold_data = epoch_df[epoch_df['Fold'] == fold]
    plt.plot(fold_data['Epoch'], fold_data['Validation F1-score'], marker='o', label=f'Fold {fold}')

plt.title('Validation F1-score by Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1-score')
plt.legend()
plt.grid(True)
plt.show()
