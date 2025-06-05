#!/usr/bin/env python3
"""
P-NET Model Version 2: Enhanced Scikit-learn Model (Paper Aligned)

This script implements an enhanced version of the P-NET model incorporating
methodological aspects from the Elmarakeby et al. (2021) P-NET paper:
- Improved data splitting (train/validation/test: 80/10/10)
- Class imbalance handling with balanced class weights
- Additional evaluation metrics (AUPRC)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Add the scripts directory to Python path to import preprocessing module
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Import the preprocessing module
import load_and_preprocess_pnet_data


def load_preprocessed_data():
    """Load preprocessed data using the preprocessing script."""
    print("Loading preprocessed data...")
    print("=" * 70)
    
    # Call the main function from preprocessing script
    mutation_df, cna_df, response_df = load_and_preprocess_pnet_data.main()
    
    print("\nData loading complete!")
    print("=" * 70)
    
    return mutation_df, cna_df, response_df


def create_feature_matrix(mutation_df, cna_df):
    """
    Combine mutation and CNA data into a single feature matrix.
    
    Args:
        mutation_df: DataFrame with mutation data (samples x genes)
        cna_df: DataFrame with CNA data (samples x genes)
    
    Returns:
        X: Combined feature matrix with prefixed column names
    """
    print("\n2. Creating Combined Feature Matrix...")
    print("-" * 50)
    
    # Ensure both DataFrames have the same sample order
    assert list(mutation_df.index) == list(cna_df.index), "Sample order mismatch"
    
    # Prefix column names to avoid duplicates
    mutation_df_prefixed = mutation_df.add_prefix('mut_')
    cna_df_prefixed = cna_df.add_prefix('cna_')
    
    # Concatenate horizontally
    X = pd.concat([mutation_df_prefixed, cna_df_prefixed], axis=1)
    
    print(f"  Mutation features: {mutation_df_prefixed.shape[1]}")
    print(f"  CNA features: {cna_df_prefixed.shape[1]}")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Combined matrix shape: {X.shape}")
    
    # Check for NaN values
    nan_count = X.isna().sum().sum()
    print(f"  NaN values in combined matrix: {nan_count}")
    
    return X


def split_data_stratified(X, response_df):
    """
    Split data into train/validation/test sets (80/10/10) with stratification.
    
    Args:
        X: Feature matrix
        response_df: DataFrame with response data
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Split data
    """
    print("\n3. Splitting Data (80/10/10 with Stratification)...")
    print("-" * 50)
    
    # Extract target variable
    y = response_df['response'].values
    print(f"  Total samples: {len(y)}")
    print(f"  Class distribution: {pd.Series(y).value_counts(normalize=True).to_dict()}")
    
    # First split: separate test set (10%) from train+val (90%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    
    # Second split: separate train (80% of total) and val (10% of total)
    # Since temp has 90% of data, we need 1/9 of temp to be validation (10% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=1/9, stratify=y_temp, random_state=42
    )
    
    # Print shapes and distributions
    print(f"\n  Dataset sizes:")
    print(f"    Training:   {len(y_train)} samples ({len(y_train)/len(y)*100:.1f}%)")
    print(f"    Validation: {len(y_val)} samples ({len(y_val)/len(y)*100:.1f}%)")
    print(f"    Test:       {len(y_test)} samples ({len(y_test)/len(y)*100:.1f}%)")
    
    print(f"\n  Class distributions:")
    print(f"    Training:   {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
    print(f"    Validation: {pd.Series(y_val).value_counts(normalize=True).to_dict()}")
    print(f"    Test:       {pd.Series(y_test).value_counts(normalize=True).to_dict()}")
    
    print(f"\n  Feature shapes:")
    print(f"    X_train: {X_train.shape}")
    print(f"    X_val:   {X_val.shape}")
    print(f"    X_test:  {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model_pipeline():
    """
    Create a Pipeline with StandardScaler and LogisticRegression.
    
    Returns:
        pipeline: Sklearn Pipeline object
    """
    print("\n4. Creating Model Pipeline...")
    print("-" * 50)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            class_weight='balanced',  # Key change for class imbalance
            max_iter=1000,
            random_state=42
        ))
    ])
    
    print("  Pipeline components:")
    print("    1. StandardScaler")
    print("    2. LogisticRegression")
    print("\n  Logistic Regression parameters:")
    print("    - penalty: l2")
    print("    - C: 1.0")
    print("    - solver: liblinear")
    print("    - class_weight: balanced (addresses class imbalance)")
    print("    - max_iter: 1000")
    print("    - random_state: 42")
    
    return pipeline


def evaluate_model(pipeline, X, y, dataset_name):
    """
    Evaluate the model on a given dataset.
    
    Args:
        pipeline: Trained sklearn Pipeline
        X: Features
        y: True labels
        dataset_name: Name of the dataset (e.g., "Training", "Validation", "Test")
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n{dataset_name} Set Evaluation:")
    print("-" * 40)
    
    # Make predictions
    y_pred = pipeline.predict(X)
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='binary'),
        'recall': recall_score(y, y_pred, average='binary'),
        'f1_score': f1_score(y, y_pred, average='binary'),
        'auc_roc': roc_auc_score(y, y_pred_proba),
        'auc_prc': average_precision_score(y, y_pred_proba)  # AUPRC
    }
    
    # Print metrics
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  AUC-PRC:   {metrics['auc_prc']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"     [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    
    # Classification Report
    print(f"\n  Classification Report:")
    report = classification_report(y, y_pred, 
                                 target_names=['Primary', 'Metastatic'])
    # Add indentation manually
    indented_report = '\n'.join('    ' + line for line in report.split('\n'))
    print(indented_report)
    
    return metrics


def main():
    """Main function to run the enhanced P-NET model pipeline."""
    print("\n" + "=" * 70)
    print("P-NET MODEL VERSION 2 - ENHANCED SCIKIT-LEARN (PAPER ALIGNED)")
    print("=" * 70)
    
    try:
        # Step 1: Load preprocessed data
        mutation_df, cna_df, response_df = load_preprocessed_data()
        
        # Step 2: Create combined feature matrix
        X = create_feature_matrix(mutation_df, cna_df)
        
        # Step 3: Split data (80/10/10 with stratification)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(X, response_df)
        
        # Step 4: Create model pipeline
        pipeline = create_model_pipeline()
        
        # Step 5: Train model
        print("\n5. Training Model...")
        print("-" * 50)
        pipeline.fit(X_train, y_train)
        print("  ✓ Model training complete!")
        
        # Step 6: Evaluate on all datasets
        print("\n6. Model Evaluation")
        print("=" * 50)
        
        # Training set evaluation
        train_metrics = evaluate_model(pipeline, X_train, y_train, "Training")
        
        # Validation set evaluation
        val_metrics = evaluate_model(pipeline, X_val, y_val, "Validation")
        
        # Test set evaluation
        test_metrics = evaluate_model(pipeline, X_test, y_test, "Test")
        
        # Summary comparison
        print("\n" + "=" * 70)
        print("SUMMARY - PERFORMANCE ACROSS DATASETS")
        print("=" * 70)
        print(f"{'Metric':<12} {'Training':>10} {'Validation':>12} {'Test':>10}")
        print("-" * 50)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_prc']:
            print(f"{metric:<12} {train_metrics[metric]:>10.4f} "
                  f"{val_metrics[metric]:>12.4f} {test_metrics[metric]:>10.4f}")
        
        print("\n" + "=" * 70)
        print("P-NET MODEL V2 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return pipeline, train_metrics, val_metrics, test_metrics
        
    except Exception as e:
        print(f"\n✗ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    pipeline, train_metrics, val_metrics, test_metrics = main()