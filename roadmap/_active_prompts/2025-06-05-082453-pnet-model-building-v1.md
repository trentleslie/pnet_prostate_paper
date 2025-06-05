# Task: P-NET Model Building - Version 1 (Baseline Genomic Model)

**Source Prompt Reference:** This task is defined by the prompt: `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-05-082453-pnet-model-building-v1.md`

## 1. Task Objective
To develop a Python script that implements a baseline version of the P-NET (Prostate Network) model. This version will use the preprocessed somatic mutation and copy number alteration (CNA) data to predict patient response (primary vs. metastatic tumor). The model should operate in a 'genomic-data-only' mode, consistent with an `ignore_missing_histology=True` approach.

## 2. Prerequisites
- [X] Required Python script for data preprocessing exists:
    - `/procedure/pnet_prostate_paper/scripts/load_and_preprocess_pnet_data.py` (This script should be callable or its functions importable to provide the three processed DataFrames: `mutation_df_processed`, `cna_df_processed`, `response_df_processed`).
- [X] Required permissions: Read access to the preprocessing script and execute permissions if it's run as a standalone script to generate data.
- [X] Required dependencies: Python 3.x, pandas, scikit-learn.
    - Installation (if needed): `pip install pandas scikit-learn` or `poetry add pandas scikit-learn`
- [X] Environment state: A Python environment where pandas and scikit-learn can be imported and run.

## 3. Context from Previous Attempts (if applicable)
- **Previous attempt timestamp:** N/A (This is the first attempt for P-NET model building)
- **Issues encountered:** N/A
- **Partial successes:** N/A
- **Recommended modifications:** N/A

## 4. Task Decomposition
Break this task into the following verifiable subtasks:
1.  **Load Preprocessed Data:**
    *   Execute or import from `/procedure/pnet_prostate_paper/scripts/load_and_preprocess_pnet_data.py` to obtain `mutation_df_processed`, `cna_df_processed`, and `response_df_processed` pandas DataFrames.
    *   Validation: Confirm DataFrames are loaded with expected shapes (mutation/CNA: 1012 samples x 9205 genes; response: 1012 samples x 1 column).
2.  **Feature Engineering & Combination:**
    *   Create a combined feature matrix `X` from `mutation_df_processed` and `cna_df_processed`.
    *   Ensure unique column names in `X` (e.g., by prefixing original gene names with `mut_` and `cna_`).
    *   Prepare the target variable `y` from the 'response' column of `response_df_processed` (ensure it's a 1D array or Series).
    *   Validation: Print shape of `X` and `y`; check for NaNs in `X` and `y`.
3.  **Data Splitting:**
    *   Split `X` and `y` into training and testing sets (e.g., 80% train, 20% test).
    *   Use stratification based on `y` to ensure proportional class representation in splits.
    *   Set a `random_state` for reproducibility.
    *   Validation: Print shapes of `X_train`, `X_test`, `y_train`, `y_test`.
4.  **Model Selection & Definition (Baseline):**
    *   Select and define a baseline classification model from `scikit-learn` (e.g., Logistic Regression with L2 regularization or Random Forest Classifier).
    *   Use default hyperparameters for this first version, or sensible starting values (e.g., `C=1.0` for Logistic Regression, `n_estimators=100` for Random Forest).
    *   Set a `random_state` for the model if applicable for reproducibility.
    *   Validation: Confirm model object is instantiated.
5.  **Model Training:**
    *   Train the selected model using `X_train` and `y_train`.
    *   Validation: Confirm model training completes without errors.
6.  **Model Evaluation:**
    *   Make predictions on `X_test`.
    *   Calculate and print the following classification metrics: Accuracy, Precision, Recall, F1-score, and AUC-ROC.
    *   If possible, print the confusion matrix.
    *   Validation: Metrics are successfully calculated and printed.
7.  **Code Structuring:**
    *   Organize the entire process into a well-commented Python script (e.g., `pnet_model_v1.py`).
    *   The script should be runnable from the command line.
    *   Validation: Script runs end-to-end and produces the specified outputs.

## 5. Implementation Requirements
- **Input data source:** The script `/procedure/pnet_prostate_paper/scripts/load_and_preprocess_pnet_data.py` should be used to provide the three DataFrames.
- **Expected outputs:**
    1.  A Python script (e.g., `pnet_model_v1.py`) containing the full model building pipeline.
    2.  Printed output to console: shapes of data at various stages, selected model and its parameters, and all specified evaluation metrics for the test set.
- **Code standards:** Use pandas for data handling, scikit-learn for modeling. Clear comments, reproducible `random_state` usage, logical flow.
- **Validation requirements:** As specified in each subtask. The final script should run without errors and produce all requested printouts.

## 6. Error Recovery Instructions
If you encounter errors during execution:
- **Data Loading/Access Errors:**
    - Ensure `/procedure/pnet_prostate_paper/scripts/load_and_preprocess_pnet_data.py` runs correctly and returns the expected DataFrames. Debug that script if necessary.
    - Classification: RETRY_WITH_MODIFICATIONS (after fixing data loading).
- **Dependency Errors (e.g., `ModuleNotFoundError` for `sklearn`):
    - Instruct user to install: `pip install scikit-learn` or `poetry add scikit-learn`.
    - Classification: RETRY_WITH_MODIFICATIONS (after installation).
- **Model Training Errors (e.g., `ValueError: Input contains NaN, infinity or a value too large`):
    - Re-check `X_train` for NaNs or problematic values. Ensure preprocessing handled these.
    - Consider adding a simple imputer or scaler from scikit-learn if appropriate, though preprocessed data should be clean.
    - Classification: RETRY_WITH_MODIFICATIONS.
- **Metric Calculation Errors:**
    - Ensure predictions and true labels are in the correct format for metric functions.
    - For AUC-ROC, ensure probability predictions are used if required by the function.
    - Classification: RETRY_WITH_MODIFICATIONS.

For each error type, indicate in your feedback:
- Error classification: [RETRY_WITH_MODIFICATIONS | ESCALATE_TO_USER | REQUIRE_DIFFERENT_APPROACH]
- Specific changes needed for retry (if applicable)
- Confidence level in proposed solution

## 7. Success Criteria and Validation
Task is complete when:
- [ ] A Python script for P-NET model v1 is created and successfully executes end-to-end.
- [ ] The script correctly loads or generates the three preprocessed DataFrames.
- [ ] Features from mutation and CNA data are successfully combined.
- [ ] Data is split into training and testing sets with stratification.
- [ ] A baseline scikit-learn classification model is trained on the training data without errors.
- [ ] Predictions are made on the test set.
- [ ] Accuracy, Precision, Recall, F1-score, and AUC-ROC are calculated and printed for the test set.
- [ ] A confusion matrix for the test set is printed (if feasible).

## 8. Feedback Requirements
Create a detailed Markdown feedback file at:
`[PROJECT_ROOT]/roadmap/_active_prompts/feedback/YYYY-MM-DD-HHMMSS-feedback-pnet-model-building-v1.md`
(Replace `[PROJECT_ROOT]` with the actual project root, and `YYYY-MM-DD-HHMMSS` with the execution timestamp in UTC).

**Mandatory Feedback Sections:**
- **Execution Status:** [COMPLETE_SUCCESS | PARTIAL_SUCCESS | FAILED_WITH_RECOVERY_OPTIONS | FAILED_NEEDS_ESCALATION]
- **Completed Subtasks:** [List subtasks from section 4 that were successfully completed]
- **Issues Encountered:** [Detailed error messages, tracebacks, and context for any issues]
- **Model Chosen & Parameters:** [Specify the scikit-learn model used and its key parameters]
- **Evaluation Metrics (Test Set):** [Report Accuracy, Precision, Recall, F1-score, AUC-ROC, and Confusion Matrix]
- **Path to Created Script:** [Full absolute path to the generated Python model script]
- **Next Action Recommendation:** [e.g., Hyperparameter tuning, Try different models, Feature importance analysis, Escalate with questions]
- **Confidence Assessment:** [Your confidence in the model's initial performance and correctness of implementation (High/Medium/Low) and why]
- **Environment Changes:** [Any files created/modified (other than feedback/model script), dependencies installed, etc.]
- **Lessons Learned:** [Any insights gained during model building]
- **Output Snippets:** Include printed metrics and model info as part of the feedback or confirm they were printed to stdout.
