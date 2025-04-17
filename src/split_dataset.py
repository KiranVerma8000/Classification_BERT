import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

path = '/Users/mitul/Documents/study/sem 4/DSSE/Assignmets/Assignments 3/assignment3_repo/ds4se3-group6'

class DatasetHandler:
    def __init__(self, filepath):
        self.df = pd.read_excel(filepath)
        self.label_encoder = LabelEncoder()

        self.df['Synthesis'] = self.df['Purpose'].apply(lambda x: 1 if 'synthesis' in str(x).lower() else 0)
        self.df['Evaluation'] = self.df['Purpose'].apply(lambda x: 1 if 'evaluation' in str(x).lower() else 0)
        self.df['Analysis'] = self.df['Purpose'].apply(lambda x: 1 if 'analysis' in str(x).lower() else 0)
        self.df['Is_Architectural'] = self.df['Post_Type'].apply(lambda row: 1 if'Architectural'in str(row)else 0,)
       
        self.X = self.df[['preprocessed_Text', 'Synthesis', 'Evaluation', 'Analysis']]
        self.y = self.df['Is_Architectural']

    def initial_split(self):
    
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=0.1, random_state=42, stratify=self.y)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val)  # 0.1111 * 0.9 ≈ 0.1
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def perform_k_fold_split(self, X, y, k=10):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        folds = []
        for train_val_index, test_index in skf.split(X, y):
            X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
            y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val)  # 0.1111 * 0.9 ≈ 0.1
            folds.append((X_train, X_val, X_test, y_train, y_val, y_test))
        return folds

    def save_splits(self, X_train, X_val, X_test, y_train, y_val, y_test):
        
        train_df = X_train.copy()
        train_df['Is_Architectural'] = y_train.values
        
        val_df = X_val.copy()
        val_df['Is_Architectural'] = y_val.values
        
        test_df = X_test.copy()
        test_df['Is_Architectural'] = y_test.values
        
        train_df.to_excel(f'{path}/datasets/split_dataset/train_set.xlsx', index=False)
        val_df.to_excel(f'{path}/datasets/split_dataset/validation_set.xlsx', index=False)
        test_df.to_excel(f'{path}/datasets/split_dataset/test_set.xlsx', index=False)
    
    def save_folds(self, folds):
        # Handle the data saving for each fold
        for i, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(folds):
            train_fold = X_train.copy()
            train_fold['Is_Architectural'] = y_train.values
            
            val_fold = X_val.copy()
            val_fold['Is_Architectural'] = y_val.values
            
            test_fold = X_test.copy()
            test_fold['Is_Architectural'] = y_test.values
            
            train_fold.to_excel(f'{path}/datasets/split_dataset/k-cross/train_fold/train_fold_{i+1}.xlsx', index=False)
            val_fold.to_excel(f'{path}/datasets/split_dataset/k-cross/validation_fold/validation_fold_{i+1}.xlsx', index=False)
            test_fold.to_excel(f'{path}/datasets/split_dataset/k-cross/test_fold/test_fold_{i+1}.xlsx', index=False)

if __name__ == "__main__":
    handler = DatasetHandler(path+'/datasets/preprocessed_data.xlsx')
    X_train, X_val, X_test, y_train, y_val, y_test = handler.initial_split()
    handler.save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    folds = handler.perform_k_fold_split(handler.X, handler.y)
    handler.save_folds(folds)
