# Task: Debug and Refactor SKCM Tumor Purity Prediction Script

**Source Prompt Reference:** This task is defined by the prompt: `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-05-074500-debug-skcm-purity-script.md` (this file)

## 1. Task Objective
The primary goal is to make the Python script `/procedure/pnet_prostate_paper/notebooks/SKCM_pnet_tf2.py` (note: original filename might be `SKCM_purity_tf2.py`, please confirm and use the correct one) fully runnable and functional for predicting SKCM (Skin Cutaneous Melanoma) tumor purity using a P-NET model with TensorFlow 2.x. This involves resolving missing function definitions, correcting data loading paths, and ensuring overall script coherence and TF2 compatibility.

## 2. Prerequisites
-   [ ] Access to the project codebase at `/procedure/pnet_prostate_paper/`.
-   [ ] Familiarity with the P-NET architecture (refer to `/procedure/pnet_prostate_paper/paper.txt` and existing model builders in `/procedure/pnet_prostate_paper/model/builders/`).
-   [ ] Understanding of TensorFlow 2.x.
-   [ ] Project dependencies (Python 3.11, TensorFlow, pandas, numpy, scikit-learn, matplotlib, seaborn) should be installed and available in the execution environment. The script `SKCM_purity_tf2.py` uses `sys.path.insert(0, '/procedure/pnet_prostate_paper')` to attempt to make project modules importable.

## 3. Context from Previous Attempts (if applicable)
-   This is the first systematic attempt to debug this specific script via a detailed prompt.
-   The script `/procedure/pnet_prostate_paper/notebooks/SKCM_purity_tf2.py` is an adaptation of a PyTorch example notebook to TensorFlow 2.x.
-   Key known issues:
    *   The function `build_pnet_regression_model` is called but not defined within the script.
    *   Data loading functions attempt to fetch data from GitHub, which might be unreliable or outdated; local data usage is preferred.

## 4. Task Decomposition
Break this task into the following verifiable subtasks:

1.  **Locate/Implement `build_pnet_regression_model`:**
    *   **Subtask 1.1:** Search within `/procedure/pnet_prostate_paper/model/builders/` (e.g., `prostate_models.py`, `generic_builders.py` if it exists) for an existing P-NET model builder suitable for regression or adaptable for it.
    *   **Subtask 1.2:** If a suitable builder is found, ensure it is TensorFlow 2.x compatible. Adapt it to meet the requirements of a regression model:
        *   The final output layer should have 1 unit.
        *   The activation function for the output layer should be linear (or appropriate for regression, possibly sigmoid if purity is strictly 0-1 and scaled). The script seems to expect purity values between 0 and 1.
        *   The loss function used during compilation with this model will be Mean Squared Error (MSE) or the custom `WeightedMSELoss` defined in the script.
    *   **Subtask 1.3:** If no directly suitable builder exists, adapt an existing P-NET classification builder (e.g., `build_pnet` from `model.builders.prostate_models.py`, if it's generic enough beyond prostate cancer specifics). This will involve modifying the output layer and ensuring the architecture can accept the `config` dictionary parameters used in `SKCM_purity_tf2.py` (e.g., `n_hidden_layers`, `activation`, `dropout`, `w_reg`).
    *   **Subtask 1.4:** The `build_pnet_regression_model` function should take `n_features` (number of input features), `n_genes` (number of available genes for the gene layer, potentially `len(available_genes)` from the script), and the `config` dictionary as input. It should return a compiled or uncompiled Keras model.
    *   **Subtask 1.5:** Place the implemented or adapted `build_pnet_regression_model` function in an appropriate existing builder file (e.g., `model/builders/generic_builders.py` - create if it doesn't exist) or within `model/builders/prostate_models.py` if it's a minor adaptation of an existing function there. Ensure it can be imported correctly by `SKCM_purity_tf2.py`.

2.  **Update Data Loading Functions:**
    *   **Subtask 2.1:** Modify `load_tcga_skcm_data()`:
        *   Change data loading to prioritize local files for SKCM RNA-seq and CNA data.
        *   RNA-seq data: `/procedure/pnet_prostate_paper/data/_database/skcm_tcga_pan_can_atlas_2018/data_RNA_Seq_v2_expression_median.txt`
        *   CNA data: `/procedure/pnet_prostate_paper/data/_database/skcm_tcga_pan_can_atlas_2018/data_CNA.txt`
        *   Ensure that data transformations (e.g., `rna.drop(['Entrez_Gene_Id'], errors='ignore').T`) are compatible with the format of these local files.
        *   Retain the synthetic data generation as a fallback if local files are not found or fail to load.
    *   **Subtask 2.2:** Modify `load_tumor_purity_data(rna_samples)`:
        *   Prioritize loading tumor purity data from local files. Check for `TCGA_mastercalls.abs_tables_JSedit.fixed.txt` or `ABSOLUTE_scores_SKCM.csv` within `/procedure/pnet_prostate_paper/data/_database/skcm_tcga_pan_can_atlas_2018/` or `/procedure/pnet_prostate_paper/data/_database/auxiliary_data/`. Adjust path as necessary.
        *   Ensure it correctly aligns purity data with the provided `rna_samples` index.
        *   Retain synthetic purity data generation as a fallback.
    *   **Subtask 2.3:** Modify `load_cancer_genes()`:
        *   Prioritize loading the cancer gene list from a local file, likely `cancer_gene_census.csv`. Check for this file in `/procedure/pnet_prostate_paper/data/_database/pathway_data/` or `/procedure/pnet_prostate_paper/data/_database/auxiliary_data/`. Adjust path as necessary.
        *   Retain synthetic gene list generation as a fallback.

3.  **Verify Imports and Script Execution:**
    *   **Subtask 3.1:** Ensure that the `build_pnet_regression_model` (from Subtask 1) and any other necessary project modules are correctly imported in `SKCM_purity_tf2.py`.
    *   **Subtask 3.2:** Execute the entire `SKCM_purity_tf2.py` script from the `/procedure/pnet_prostate_paper/notebooks/` directory (or project root, ensuring paths are correct).
    *   **Subtask 3.3:** Confirm that the script runs end-to-end without Python errors or TensorFlow/Keras errors. This includes data loading, preprocessing, model building, training, evaluation, and generation of plots.
    *   **Subtask 3.4:** Pay attention to shapes of data tensors and model layers to ensure compatibility.

## 5. Implementation Requirements
-   **Input files/data:**
    *   Primary script: `/procedure/pnet_prostate_paper/notebooks/SKCM_purity_tf2.py`
    *   Project model builders: `/procedure/pnet_prostate_paper/model/builders/`
    *   Local data sources:
        *   RNA: `/procedure/pnet_prostate_paper/data/_database/skcm_tcga_pan_can_atlas_2018/data_RNA_Seq_v2_expression_median.txt`
        *   CNA: `/procedure/pnet_prostate_paper/data/_database/skcm_tcga_pan_can_atlas_2018/data_CNA.txt`
        *   Purity (example): `/procedure/pnet_prostate_paper/data/_database/auxiliary_data/TCGA_mastercalls.abs_tables_JSedit.fixed.txt` (verify actual file and location)
        *   Cancer Genes (example): `/procedure/pnet_prostate_paper/data/_database/pathway_data/cancer_gene_census.csv` (verify actual file and location)
    *   Reference: `/procedure/pnet_prostate_paper/paper.txt` for P-NET context.
    *   Reference: `/procedure/pnet_prostate_paper/roadmap/technical_notes/pnet_refactor.md` for TF2 migration patterns.
-   **Expected outputs:**
    *   A modified, runnable version of `/procedure/pnet_prostate_paper/notebooks/SKCM_purity_tf2.py`.
    *   If `build_pnet_regression_model` is newly created or significantly adapted, provide the code for this function and indicate where it should be saved (e.g., `model/builders/generic_builders.py`).
    *   A brief report on changes made, especially concerning data paths and the model builder.
-   **Code standards:**
    *   Follow existing code style in the project.
    *   Ensure TensorFlow 2.x best practices.
    *   Add comments to explain significant changes or complex logic.
-   **Validation requirements:**
    *   The script must execute without errors.
    *   Data should be loaded correctly from local sources if available.
    *   The P-NET model should train, and evaluation metrics should be produced.
    *   Plots should be generated as in the original script.

## 6. Error Recovery Instructions
-   **Permission/Tool Errors:** If file access errors occur, note the paths and permissions likely needed.
-   **Dependency Errors:** If Python packages are missing, list them. The script already imports common libraries; project-specific imports are the main concern.
-   **Configuration Errors:** If the `config` dictionary in the script is incompatible with the P-NET builder, document the discrepancies and suggest modifications to the `config` or the builder's parameter handling.
-   **Logic/Implementation Errors:**
    *   If adapting an existing P-NET builder is too complex or leads to many errors, clearly document the issues.
    *   If local data formats are significantly different from what the script expects, detail the necessary preprocessing adjustments.
    *   If TensorFlow errors occur, provide the full traceback and try to identify the cause (e.g., shape mismatches, deprecated TF1.x usage in a P-NET component).
-   For each error type, indicate in your feedback:
    -   Error classification: [RETRY_WITH_MODIFICATIONS | ESCALATE_TO_USER | REQUIRE_DIFFERENT_APPROACH]
    -   Specific changes needed for retry (if applicable)
    -   Confidence level in proposed solution

## 7. Success Criteria and Validation
Task is complete when:
-   [ ] The `build_pnet_regression_model` function is correctly implemented/adapted and integrated.
-   [ ] Data loading functions in `SKCM_purity_tf2.py` are updated to prioritize local data sources from `/procedure/pnet_prostate_paper/data/_database/`.
-   [ ] The script `/procedure/pnet_prostate_paper/notebooks/SKCM_purity_tf2.py` runs successfully from start to finish without errors.
-   [ ] The script produces meaningful output (training logs, evaluation metrics, plots).
-   [ ] All modifications are documented.

## 8. Feedback Requirements
Create a detailed Markdown feedback file at:
`[PROJECT_ROOT]/roadmap/_active_prompts/feedback/YYYY-MM-DD-HHMMSS-feedback-debug-skcm-purity-script.md`

**Mandatory Feedback Sections:**
-   **Execution Status:** [COMPLETE_SUCCESS | PARTIAL_SUCCESS | FAILED_WITH_RECOVERY_OPTIONS | FAILED_NEEDS_ESCALATION]
-   **Completed Subtasks:** Checklist of what was accomplished from Section 4.
-   **Issues Encountered:** Detailed error descriptions, tracebacks, and context.
-   **Changes Made:**
    *   Summary of modifications to `SKCM_purity_tf2.py`.
    *   Code for the `build_pnet_regression_model` function and its intended file location.
    *   Details of any changes to data paths or preprocessing logic.
-   **Next Action Recommendation:** Specific follow-up needed (e.g., "Review implemented builder function", "Verify data paths for X").
-   **Confidence Assessment:** Quality of the fix, testing coverage, potential remaining risks.
-   **Environment Changes:** Any assumptions made about the environment or project structure.
-   **Lessons Learned:** Patterns that worked or should be avoided for similar tasks.
