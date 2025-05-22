## Status

**Current Stage:** 2_inprogress

## Progress Log

*   **2025-05-22 (approx):**
    *   Moved from `1_planning` to `2_inprogress`.
    *   Analyzed `_logs.zip` and `_database.zip` contents provided by USER.
    *   Key Insight: Discovered an example `_params.yml` file (`P-net_ALL_params.yml`) saved within the historical logs. This file provides the canonical structure for `model_params` and `data_params`, including nested structures.
    *   Created `template_params.yml` in `/procedure/pnet_prostate_paper/test_configs/mock_params/` based on the structure of the discovered `P-net_ALL_params.yml`. This template includes detailed comments and placeholders.
    *   Path A (Investigation & Recovery) is now largely superseded by the findings from log analysis, which provided a concrete example of a `_params.yml`.
*   **Next Steps (Path B - Mock/Template Parameter File System):**
    *   Create a `README.md` for the `/procedure/pnet_prostate_paper/test_configs/` directory.
    *   Develop specific mock parameter files (e.g., `mock_basic_nn_params.yml`, `mock_nn_with_gradient_importance_params.yml`) using `template_params.yml`.
    *   **Data File Path Strategy for Mock Configurations:**
        *   **Gene List:** Use `/procedure/pnet_prostate_paper/data/_database/genes/tcga_prostate_expressed_genes_and_cancer_genes.csv`.
        *   **Response/Outcome File:** Use `/procedure/pnet_prostate_paper/data/_database/prostate/processed/response_paper.csv`.
        *   **Sample Split Definition:** Use a small split file like `/procedure/pnet_prostate_paper/data/_database/prostate/splits/test_set.csv` to define the cohort of samples for mock testing.
        *   **Main Data Matrices (Mutation, CNV, Expression):** For initial mock parameter file creation and testing of loading logic, these will point to the full-sized files within `/procedure/pnet_prostate_paper/data/_database/prostate/processed/` (e.g., `P1000_final_analysis_set_cross_important_only.csv`, `P1000_data_CNA_paper.csv`, `P1000_data_tpm.csv`).
    *   **Sub-task: Create True Minimal Test Dataset:** Add a task to this FP (or create a new FP) to generate a self-contained, minimal dataset (subsetted data matrices, gene list, response, and split files) in a new `/procedure/pnet_prostate_paper/test_data/minimal_prostate_set/` directory. This will involve scripting the subsetting process. Create a dedicated mock `_params.yml` for this minimal set.

# Design: Strategy for Missing _params.yml Files / Test Data Generation

## 1. Overview
This design addresses the absence of `*_params.yml` files. The primary path is investigation and recovery. If that fails, a secondary path involves creating a robust system of mock parameter files and potentially a simplified test model setup.

## 2. Path A: Investigation & Recovery

*   **Sources to Check:**
    *   Other Git branches (local or remote).
    *   Previous commits (though `_logs/` are often gitignored).
    *   Team members' local machines.
    *   Any shared storage or backups if they exist for the project.
*   **Tools:** Standard Git commands, file system search.

## 3. Path B: Mock/Template Parameter File System (If Recovery Fails)

*   **Location of Mock Files:** A new directory, e.g., `/procedure/pnet_prostate_paper/test_configs/mock_params/`, will house various example `*_params.yml` files.
*   **Schema Definition:**
    *   Analyze the structure of `model_params_dict` in `model_factory.py` and `DataModelLoader.load_model` in `utils/loading_utils.py` to understand all required and optional fields for `*_params.yml`.
    *   Define a baseline `template_params.yml` with placeholders and comments explaining each field.
*   **Example Mock Files:** Create several mock files for common scenarios:
    *   `mock_basic_nn_params.yml` (simple neural network).
    *   `mock_nn_with_gradient_importance_params.yml` (for testing `GradientCheckpoint` with `feature_importance='gradient'`).
    *   `mock_nn_with_random_importance_params.yml` (for `feature_importance='random'`).
    *   `mock_nn_with_custom_gradient_fn_params.yml` (for `feature_importance` as a callable).
*   **Loading Mechanism Adaptation (Minor, if any):**
    *   The existing `DataModelLoader` should ideally work with these mock files if they adhere to the expected path structure (e.g., `/procedure/pnet_prostate_paper/test_configs/mock_params/experiment_name/model_name_params.yml`).
    *   Alternatively, scripts might need a flag or configuration to point to this `test_configs/` directory instead of `_logs/`.

## 4. Path C: Minimal Test Model & Data (Optional Add-on to Path B)

*   **Purpose:** If setting up full mock parameter files for existing complex models is too cumbersome for basic callback/mechanics testing.
*   **Components:**
    *   A very simple Keras model definition (e.g., 1-2 layers) within a test script.
    *   Generation of dummy NumPy data (e.g., `np.random.rand()`) for input and target.
*   **Usage:** Test scripts can instantiate this simple model, compile it, and run a few training steps to trigger callbacks like `GradientCheckpoint`, directly providing parameters like `feature_importance` programmatically.

## 5. Documentation

*   A `README.md` within `/procedure/pnet_prostate_paper/test_configs/` explaining how to use the mock parameter files or the minimal test model setup.
