#!/usr/bin/env python3
"""
Load and preprocess P-NET cancer datasets.

This script loads and preprocesses three key prostate cancer datasets:
- Somatic mutation data
- Copy number alteration (CNA) data
- Clinical response data

The output will be three cleaned, aligned, and validated pandas DataFrames.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Define paths to data files
DATA_DIR = Path("/procedure/pnet_prostate_paper/data/_database/prostate/processed")
MUTATION_FILE = DATA_DIR / "P1000_final_analysis_set_cross_important_only_plus_hotspots.csv"
CNA_FILE = DATA_DIR / "P1000_data_CNA_paper.csv"
RESPONSE_FILE = DATA_DIR / "response_paper.csv"


def main():
    """Main function to load and preprocess all datasets."""
    print("Starting P-NET data loading and preprocessing...")
    print("=" * 70)
    
    # Step 1: Load Somatic Mutation Data
    print("\n1. Loading Somatic Mutation Data...")
    print("-" * 50)
    try:
        mutation_df = pd.read_csv(MUTATION_FILE, index_col=0)
        print(f"✓ Loaded mutation data from: {MUTATION_FILE}")
        print(f"  Initial shape: {mutation_df.shape}")
        print(f"  First 5 rows preview:")
        print(mutation_df.head())
        
        # Handle missing values by filling with 0.0
        mutation_df = mutation_df.fillna(0.0)
        print(f"\n  Filled NaN values with 0.0")
        
        # Ensure all data is numeric
        mutation_df = mutation_df.astype(float)
        print(f"  Converted all values to float")
        
        # Validation
        print(f"\n  Validation:")
        print(f"    - Shape after processing: {mutation_df.shape}")
        print(f"    - Contains NaN: {mutation_df.isna().sum().sum() > 0}")
        print(f"    - Data types: {mutation_df.dtypes.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"✗ Error loading mutation data: {e}")
        raise
    
    # Step 2: Load CNA Data
    print("\n2. Loading CNA Data...")
    print("-" * 50)
    try:
        cna_df = pd.read_csv(CNA_FILE, index_col=0)
        print(f"✓ Loaded CNA data from: {CNA_FILE}")
        print(f"  Initial shape: {cna_df.shape}")
        print(f"  First 5 rows preview:")
        print(cna_df.head())
        
        # Ensure all data is numeric
        cna_df = cna_df.astype(float)
        print(f"\n  Converted all values to float")
        
        # Validation
        print(f"\n  Validation:")
        print(f"    - Shape: {cna_df.shape}")
        print(f"    - Data types: {cna_df.dtypes.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"✗ Error loading CNA data: {e}")
        raise
    
    # Step 3: Load Response Data
    print("\n3. Loading Response Data...")
    print("-" * 50)
    try:
        response_df = pd.read_csv(RESPONSE_FILE)
        # Set 'id' column as index (the file has 'id' not 'Sample')
        response_df = response_df.set_index('id')
        print(f"✓ Loaded response data from: {RESPONSE_FILE}")
        print(f"  Initial shape: {response_df.shape}")
        print(f"  First 5 rows preview:")
        print(response_df.head())
        
        # Ensure response column is integer
        response_df['response'] = response_df['response'].astype(int)
        print(f"\n  Converted response column to integer")
        
        # Validation
        print(f"\n  Validation:")
        print(f"    - Shape: {response_df.shape}")
        print(f"    - Response column dtype: {response_df['response'].dtype}")
        print(f"    - Unique response values: {sorted(response_df['response'].unique())}")
        
    except Exception as e:
        print(f"✗ Error loading response data: {e}")
        raise
    
    # Step 4: Align Samples
    print("\n4. Aligning Samples...")
    print("-" * 50)
    try:
        # Find common samples across all three datasets
        mutation_samples = set(mutation_df.index)
        cna_samples = set(cna_df.index)
        response_samples = set(response_df.index)
        
        print(f"  Sample counts:")
        print(f"    - Mutation data: {len(mutation_samples)}")
        print(f"    - CNA data: {len(cna_samples)}")
        print(f"    - Response data: {len(response_samples)}")
        
        common_samples = mutation_samples & cna_samples & response_samples
        common_samples = sorted(list(common_samples))  # Sort for consistent order
        
        print(f"\n  Common samples found: {len(common_samples)}")
        
        # Filter all DataFrames to retain only common samples
        mutation_df = mutation_df.loc[common_samples]
        cna_df = cna_df.loc[common_samples]
        response_df = response_df.loc[common_samples]
        
        print(f"\n  Shapes after sample alignment:")
        print(f"    - Mutation data: {mutation_df.shape}")
        print(f"    - CNA data: {cna_df.shape}")
        print(f"    - Response data: {response_df.shape}")
        
        # Verify same number of rows
        assert mutation_df.shape[0] == cna_df.shape[0] == response_df.shape[0], \
            "Sample alignment failed - different number of rows"
        print(f"  ✓ All datasets have the same number of samples: {mutation_df.shape[0]}")
        
    except Exception as e:
        print(f"✗ Error aligning samples: {e}")
        raise
    
    # Step 5: Align Genes (Features)
    print("\n5. Aligning Genes...")
    print("-" * 50)
    try:
        # Find common genes between mutation and CNA data
        mutation_genes = set(mutation_df.columns)
        cna_genes = set(cna_df.columns)
        
        print(f"  Gene counts:")
        print(f"    - Mutation data: {len(mutation_genes)}")
        print(f"    - CNA data: {len(cna_genes)}")
        
        common_genes = mutation_genes & cna_genes
        common_genes = sorted(list(common_genes))  # Sort for consistent order
        
        print(f"\n  Common genes found: {len(common_genes)}")
        
        # Filter both DataFrames to retain only common genes
        mutation_df = mutation_df[common_genes]
        cna_df = cna_df[common_genes]
        
        print(f"\n  Shapes after gene alignment:")
        print(f"    - Mutation data: {mutation_df.shape}")
        print(f"    - CNA data: {cna_df.shape}")
        
        # Verify same number of columns
        assert mutation_df.shape[1] == cna_df.shape[1], \
            "Gene alignment failed - different number of columns"
        print(f"  ✓ Both datasets have the same number of genes: {mutation_df.shape[1]}")
        
    except Exception as e:
        print(f"✗ Error aligning genes: {e}")
        raise
    
    # Step 6: Final Verification and Output
    print("\n6. Final Verification...")
    print("-" * 50)
    
    # Rename to processed DataFrames
    mutation_df_processed = mutation_df
    cna_df_processed = cna_df
    response_df_processed = response_df
    
    print("Final processed DataFrames:")
    print("\nMutation DataFrame (mutation_df_processed):")
    print(f"  Shape: {mutation_df_processed.shape}")
    print(f"  Contains NaN: {mutation_df_processed.isna().sum().sum() > 0}")
    print(f"  First 5 rows:")
    print(mutation_df_processed.head())
    
    print("\nCNA DataFrame (cna_df_processed):")
    print(f"  Shape: {cna_df_processed.shape}")
    print(f"  First 5 rows:")
    print(cna_df_processed.head())
    
    print("\nResponse DataFrame (response_df_processed):")
    print(f"  Shape: {response_df_processed.shape}")
    print(f"  Response dtype: {response_df_processed['response'].dtype}")
    print(f"  First 5 rows:")
    print(response_df_processed.head())
    
    # Final validation checks
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY:")
    print("=" * 70)
    print(f"✓ All DataFrames have {mutation_df_processed.shape[0]} aligned samples")
    print(f"✓ Mutation and CNA DataFrames have {mutation_df_processed.shape[1]} aligned genes")
    print(f"✓ Mutation DataFrame contains no NaN values: {mutation_df_processed.isna().sum().sum() == 0}")
    print(f"✓ Response column is integer type: {response_df_processed['response'].dtype == 'int64'}")
    print(f"✓ All criteria met - preprocessing complete!")
    
    # Make DataFrames available globally for further use
    globals()['mutation_df_processed'] = mutation_df_processed
    globals()['cna_df_processed'] = cna_df_processed
    globals()['response_df_processed'] = response_df_processed
    
    return mutation_df_processed, cna_df_processed, response_df_processed


if __name__ == "__main__":
    mutation_df_processed, cna_df_processed, response_df_processed = main()