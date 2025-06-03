import pandas as pd
import os
import numpy as np

# --- Configuration ---
N_SAMPLES_TRAIN = 12
N_SAMPLES_VALID = 4
N_SAMPLES_TEST = 4
N_GENES = 100

BASE_OUTPUT_DIR = "/procedure/pnet_prostate_paper/test_data/minimal_prostate_set"
FULL_DATA_BASE_DIR = "/procedure/pnet_prostate_paper/data/_database"

# Define output subdirectories
GENES_OUT_DIR = os.path.join(BASE_OUTPUT_DIR, "genes")
PROCESSED_OUT_DIR = os.path.join(BASE_OUTPUT_DIR, "processed")
SPLITS_OUT_DIR = os.path.join(BASE_OUTPUT_DIR, "splits")

# Define full data paths
FULL_GENES_FILE = os.path.join(FULL_DATA_BASE_DIR, "genes", "tcga_prostate_expressed_genes_and_cancer_genes.csv")
FULL_MUT_FILE = os.path.join(FULL_DATA_BASE_DIR, "prostate", "processed", "P1000_final_analysis_set_cross_important_only.csv")
FULL_CNA_FILE = os.path.join(FULL_DATA_BASE_DIR, "prostate", "processed", "P1000_data_CNA_paper.csv")
FULL_RNA_FILE = os.path.join(FULL_DATA_BASE_DIR, "prostate", "raw_data", "outputs_p1000_rna_n=660_tpm_matrix.tsv") # Updated path to actual RNA file
FULL_RESPONSE_FILE = os.path.join(FULL_DATA_BASE_DIR, "prostate", "processed", "response_paper.csv")

FULL_TRAIN_SPLIT_FILE = os.path.join(FULL_DATA_BASE_DIR, "prostate", "splits", "training_set.csv")
FULL_VALID_SPLIT_FILE = os.path.join(FULL_DATA_BASE_DIR, "prostate", "splits", "validation_set.csv") # Assuming validation_set.csv exists
FULL_TEST_SPLIT_FILE = os.path.join(FULL_DATA_BASE_DIR, "prostate", "splits", "test_set.csv")


def create_directories():
    """Creates the necessary output directories."""
    os.makedirs(GENES_OUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_OUT_DIR, exist_ok=True)
    os.makedirs(SPLITS_OUT_DIR, exist_ok=True)
    print(f"Created directories under {BASE_OUTPUT_DIR}")

def main():
    """Main function to generate the minimal dataset."""
    create_directories()
    np.random.seed(42) # for reproducibility

    # --- 1. Load and Subset Genes ---
    print(f"Processing genes from {FULL_GENES_FILE}...")
    all_genes_df = pd.read_csv(FULL_GENES_FILE)
    # Assuming the gene identifiers are in the first column or a specific column like 'Gene_ID'
    # For this example, let's assume it's the first column if 'Gene_ID' isn't present
    gene_id_col = 'Gene_ID' if 'Gene_ID' in all_genes_df.columns else all_genes_df.columns[0]
    
    if len(all_genes_df) > N_GENES:
        selected_genes_df = all_genes_df.sample(n=N_GENES, random_state=42)
    else:
        selected_genes_df = all_genes_df
    
    minimal_genes_list = selected_genes_df[gene_id_col].tolist()
    selected_genes_df.to_csv(os.path.join(GENES_OUT_DIR, "minimal_selected_genes.csv"), index=False)
    print(f"Saved {len(minimal_genes_list)} genes to minimal_selected_genes.csv")

    # --- 2. Load Split Files and Select Samples ---
    print("Processing sample splits...")
    try:
        train_samples_df = pd.read_csv(FULL_TRAIN_SPLIT_FILE, header=None, names=['Sample_ID'])
        # Try to load validation split, if it fails, sample from train
        try:
            valid_samples_df = pd.read_csv(FULL_VALID_SPLIT_FILE, header=None, names=['Sample_ID'])
        except FileNotFoundError:
            print(f"Warning: {FULL_VALID_SPLIT_FILE} not found. Will sample validation from training set.")
            if len(train_samples_df) > N_SAMPLES_TRAIN + N_SAMPLES_VALID:
                valid_samples_df = train_samples_df.sample(n=N_SAMPLES_VALID, random_state=42)
                train_samples_df = train_samples_df.drop(valid_samples_df.index)
            else:
                valid_samples_df = train_samples_df # Use all if not enough
        test_samples_df = pd.read_csv(FULL_TEST_SPLIT_FILE, header=None, names=['Sample_ID'])
    except Exception as e:
        print(f"Error loading split files: {e}. Exiting.")
        return

    minimal_train_samples = train_samples_df['Sample_ID'].sample(n=min(N_SAMPLES_TRAIN, len(train_samples_df)), random_state=42).tolist()
    minimal_valid_samples = valid_samples_df['Sample_ID'].sample(n=min(N_SAMPLES_VALID, len(valid_samples_df)), random_state=42).tolist()
    minimal_test_samples = test_samples_df['Sample_ID'].sample(n=min(N_SAMPLES_TEST, len(test_samples_df)), random_state=42).tolist()
    
    all_minimal_samples = list(set(minimal_train_samples + minimal_valid_samples + minimal_test_samples))

    pd.DataFrame(minimal_train_samples).to_csv(os.path.join(SPLITS_OUT_DIR, "minimal_training_samples.csv"), index=False, header=False)
    pd.DataFrame(minimal_valid_samples).to_csv(os.path.join(SPLITS_OUT_DIR, "minimal_validation_samples.csv"), index=False, header=False)
    pd.DataFrame(minimal_test_samples).to_csv(os.path.join(SPLITS_OUT_DIR, "minimal_test_samples.csv"), index=False, header=False)
    print(f"Saved minimal sample splits: {len(minimal_train_samples)} train, {len(minimal_valid_samples)} valid, {len(minimal_test_samples)} test.")

    # --- 3. Load and Subset Data Matrices (MUT, CNA, RNA) & Response ---
    data_files_to_process = {
        "mut": FULL_MUT_FILE,
        "cna": FULL_CNA_FILE,
        "rna": FULL_RNA_FILE,
        "response": FULL_RESPONSE_FILE
    }

    for key, file_path in data_files_to_process.items():
        print(f"Processing {key} data from {file_path}...")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Skipping {key} data.")
            # Create an empty placeholder if a core file like response is missing, or handle as needed
            if key == "response": # Example: create dummy if response is critical
                placeholder_df = pd.DataFrame(columns=['Sample_ID', 'ResponseType'])
                placeholder_df.to_csv(os.path.join(PROCESSED_OUT_DIR, f"minimal_{key}_data.csv"), index=False)
                print(f"Created empty placeholder for {key} data.")
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Skipping {key} data.")
            continue

        # Standardize Sample ID column name (assuming it's often the first or 'Sample_ID')
        if 'Sample_ID' in df.columns:
            sample_id_col_matrix = 'Sample_ID'
        elif 'sample_id' in df.columns:
            sample_id_col_matrix = 'sample_id'
            df = df.rename(columns={'sample_id': 'Sample_ID'})
        elif df.columns[0].lower() in ['sample', 'samples', 'sampleid', 'sample_id']:
            sample_id_col_matrix = df.columns[0]
            df = df.rename(columns={sample_id_col_matrix: 'Sample_ID'})
        else:
            print(f"Could not identify sample ID column in {key} data. Skipping.")
            continue
        
        # Subset by selected samples
        subset_df = df[df['Sample_ID'].isin(all_minimal_samples)]

        # Subset by selected genes (if not response file)
        if key != "response":
            # Identify gene columns (all columns that are not 'Sample_ID')
            gene_cols_in_matrix = [col for col in subset_df.columns if col != 'Sample_ID']
            # Find intersection of available genes in matrix and our minimal_genes_list
            genes_to_keep = [gene for gene in minimal_genes_list if gene in gene_cols_in_matrix]
            
            # Ensure Sample_ID is present, plus the selected gene columns
            columns_to_select = ['Sample_ID'] + genes_to_keep
            subset_df = subset_df[columns_to_select]

        subset_df.to_csv(os.path.join(PROCESSED_OUT_DIR, f"minimal_{key}_data.csv"), index=False)
        print(f"Saved minimal {key} data with shape {subset_df.shape}")

    print("Minimal dataset creation complete.")

if __name__ == "__main__":
    main()
