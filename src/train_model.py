import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json
import os

path = '/pc2/users/m/mknasit/DS4SE/ds4se3-group6'

class StackOverflowDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(fold):
    train_df = pd.read_excel(f'{path}/datasets/split_dataset/k-cross/train_fold/train_fold_{fold}.xlsx')
    val_df = pd.read_excel(f'{path}/datasets/split_dataset/k-cross/validation_fold/validation_fold_{fold}.xlsx')
    test_df = pd.read_excel(f'{path}/datasets/split_dataset/k-cross/test_fold/test_fold_{fold}.xlsx')
    return train_df, val_df, test_df

def preprocess_data(df):
    df['label'] = df.apply(lambda row: 0 if row['Post_Type'] == 0 else
                                        (1 if row['Synthesis'] == 1 else
                                         (2 if row['Evaluation'] == 1 else
                                          3 if row['Analysis'] == 1 else -1)),  # Added an else case for safety
                           axis=1)
    return df

def ensure_string_values(df):
    df['preprocessed_Text'] = df['preprocessed_Text'].astype(str)
    if not all(isinstance(item, str) for item in df['preprocessed_Text']):
        raise ValueError("All items in 'preprocessed_Text' must be strings.")

def tokenize_data(df, tokenizer):
    ensure_string_values(df)
    encodings = tokenizer(df['preprocessed_Text'].tolist(), truncation=True, padding=True, max_length=512)
    labels = df['label'].tolist()
    return encodings, labels

def train_and_evaluate(train_df, val_df, test_df, tokenizer, learning_rate, batch_size, epochs):
    train_encodings, train_labels = tokenize_data(train_df, tokenizer)
    val_encodings, val_labels = tokenize_data(val_df, tokenizer)
    test_encodings, test_labels = tokenize_data(test_df, tokenizer)
    
    train_dataset = StackOverflowDataset(train_encodings, train_labels)
    val_dataset = StackOverflowDataset(val_encodings, val_labels)
    test_dataset = StackOverflowDataset(test_encodings, test_labels)
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    best_val_f1 = 0.0
    early_stopping_counter = 0
    patience = 2
    fold_results = []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        for batch in val_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            val_preds.extend(torch.argmax(logits, dim=1).tolist())
            val_true.extend(labels.tolist())
        
        val_accuracy = accuracy_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_true, val_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)
        
        fold_results.append({
            'epoch': epoch + 1,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Test
    test_preds, test_true = [], []
    for batch in test_loader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        test_preds.extend(torch.argmax(logits, dim=1).tolist())
        test_true.extend(labels.tolist())

    test_accuracy = accuracy_score(test_true, test_preds)
    test_precision = precision_score(test_true, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_true, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)

    # Save the model for this fold
    fold_model_save_path = f'{path}/utils/model/bert_model_fold_{fold}.bin'
    torch.save(model.state_dict(), fold_model_save_path)
    print(f"Model for fold {fold} saved to {fold_model_save_path}")

    # Save test predictions along with corresponding texts (assuming 'preprocessed_Text' is included in test_df)
    test_predictions_df = pd.DataFrame({
        'text': test_df['preprocessed_Text'].tolist(),
        'true_label': test_true,
        'predicted_label': test_preds
    })
    test_predictions_path = f'{path}/utils/results/test_predictions_fold_{fold}.xlsx'
    test_predictions_df.to_excel(test_predictions_path, index=False)
    print(f"Test predictions for fold {fold} saved to {test_predictions_path}")

    return fold_results, {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }

# Hyperparameters
learning_rate = 2e-5
batch_size = 16
epochs = 10

# Perform 10-fold cross-validation
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

all_fold_results = []

for fold in range(1, 11):
    print(f"Fold {fold}/10")
    train_df, val_df, test_df = load_data(fold)
    train_df = preprocess_data(train_df)
    val_df = preprocess_data(val_df)
    test_df = preprocess_data(test_df)
    
    try:
        fold_results, test_report = train_and_evaluate(train_df, val_df, test_df, tokenizer, learning_rate, batch_size, epochs)
        all_fold_results.append({
            'fold': fold,
            'fold_results': fold_results,
            'test_report': test_report
        })
        print(f"Results for fold {fold}: {test_report}")
    except ValueError as e:
        print(f"Error processing fold {fold}: {e}")
        continue

# Save results
results = {
    'cross_validation_results': all_fold_results
}

with open(f'{path}/utils/results/cross_validation_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Results saved to cross_validation_results.json")
