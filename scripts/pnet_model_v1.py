#!/usr/bin/env python3
"""
P-NET Model Version 1: Baseline Genomic Model

This script implements a baseline version of the P-NET (Prostate Network) model
using preprocessed somatic mutation and copy number alteration (CNA) data to
predict patient response (primary vs. metastatic tumor).

This version operates in 'genomic-data-only' mode (ignore_missing_histology=True).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
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


def prepare_data_for_modeling(X, response_df):
    """
    Prepare data for modeling: extract target variable and split data.
    
    Args:
        X: Feature matrix
        response_df: DataFrame with response data
    
    Returns:
        X_train, X_test, y_train, y_test: Split data
    """
    print("\n3. Preparing Data for Modeling...")
    print("-" * 50)
    
    # Extract target variable
    y = response_df['response'].values
    print(f"  Target variable shape: {y.shape}")
    print(f"  Target value counts: {pd.Series(y).value_counts().to_dict()}")
    print(f"  Class distribution: {pd.Series(y).value_counts(normalize=True).to_dict()}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\n  Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # Verify stratification
    train_dist = pd.Series(y_train).value_counts(normalize=True).to_dict()
    test_dist = pd.Series(y_test).value_counts(normalize=True).to_dict()
    print(f"\n  Training set class distribution: {train_dist}")
    print(f"  Test set class distribution: {test_dist}")
    
    return X_train, X_test, y_train, y_test


def create_and_train_model(X_train, y_train):
    """
    Create and train a baseline classification model.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        model: Trained model
        scaler: Fitted StandardScaler
    """
    print("\n4. Model Selection and Training...")
    print("-" * 50)
    
    # Scale features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print("  Feature scaling applied (StandardScaler)")
    
    # Create Logistic Regression model with L2 regularization
    model = LogisticRegression(
        C=1.0,  # Regularization strength (inverse)
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    print(f"\n  Model: {type(model).__name__}")
    print(f"  Parameters: {model.get_params()}")
    
    # Train the model
    print("\n  Training model...")
    model.fit(X_train_scaled, y_train)
    print("  ✓ Model training complete!")
    
    # Training set performance
    train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\n  Training accuracy: {train_acc:.4f}")
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        X_test: Test features
        y_test: Test labels
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("\n5. Model Evaluation on Test Set...")
    print("-" * 50)
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Print metrics
    print("\nClassification Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nConfusion Matrix Breakdown:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    
    # Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Primary', 'Metastatic']))
    
    return metrics


def main():
    """Main function to run the complete P-NET model pipeline."""
    print("\n" + "=" * 70)
    print("P-NET MODEL VERSION 1 - BASELINE GENOMIC MODEL")
    print("=" * 70)
    
    try:
        # Step 1: Load preprocessed data
        mutation_df, cna_df, response_df = load_preprocessed_data()
        
        # Step 2: Create combined feature matrix
        X = create_feature_matrix(mutation_df, cna_df)
        
        # Step 3: Prepare data for modeling
        X_train, X_test, y_train, y_test = prepare_data_for_modeling(X, response_df)
        
        # Step 4: Create and train model
        model, scaler = create_and_train_model(X_train, y_train)
        
        # Step 5: Evaluate model
        metrics = evaluate_model(model, scaler, X_test, y_test)
        
        print("\n" + "=" * 70)
        print("P-NET MODEL V1 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return model, scaler, metrics
        
    except Exception as e:
        print(f"\n✗ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    model, scaler, metrics = main()