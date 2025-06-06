```markdown
# Prompt: P-NET Model v2 - Enhanced Scikit-learn (Paper Aligned)

**Objective:** Develop an enhanced version of the scikit-learn based P-NET predictive model by incorporating methodological aspects from the Elmarakeby et al. (2021) P-NET paper. This version will focus on improved data splitting (train/validation/test) and class imbalance handling, using Logistic Regression as the classifier.

**Project Root:** `/procedure/pnet_prostate_paper/`

**Input Data Source:**
- Preprocessed and aligned mutation, CNA, and response data generated by the script `/procedure/pnet_prostate_paper/scripts/load_and_preprocess_pnet_data.py`.
  - `mutation_df_processed.csv`
  - `cna_df_processed.csv`
  - `response_df_processed.csv`
  (These files should be loaded from `/procedure/pnet_prostate_paper/data/_database/prostate/processed/` if the script saves them there, or directly from the output of the loading script if passed as DataFrames).

**Prerequisites:**
- The script `load_and_preprocess_pnet_data.py` must have been successfully run, and its output data (aligned mutation, CNA, and response DataFrames/CSVs) must be available.
- Familiarity with the `pnet_model_v1.py` script and its components.

**Task Decomposition:**

1.  **Load Preprocessed Data:**
    *   Load the `mutation_df_processed`, `cna_df_processed`, and `response_df_processed` datasets.
    *   Ensure sample IDs are set as indices for all DataFrames.

2.  **Feature Combination:**
    *   Combine the mutation and CNA data into a single feature matrix `X`.
    *   Ensure the samples in `X` are aligned with the response variable `y`.
    *   The response variable `y` should be a 1D array or Series from `response_df_processed['response']`.

3.  **Data Splitting (80/10/10):**
    *   Split the data (`X`, `y`) into three sets: training (80%), validation (10%), and testing (10%).
    *   **Crucially, this split must be stratified by the response variable `y` to maintain class proportions in each set.**
    *   One way to achieve this:
        *   First, split `X` and `y` into a combined training/validation set (90% of data) and a test set (10% of data). Use `train_test_split` with `test_size=0.1`, `stratify=y`, and `random_state=42`.
        *   Then, split the combined training/validation set into the final training set (8/9 or ~88.89% of this set, which is 80% of original) and the validation set (1/9 or ~11.11% of this set, which is 10% of original). Use `train_test_split` with `test_size=1/9` (or `test_size=0.111111`), `stratify` on the corresponding y subset, and `random_state=42`.
        *   Alternatively: First split 80% for training and 20% for a temporary set. Then split the temporary set 50/50 into validation and test. Ensure stratification at each step.
    *   Verify the shapes of `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`.

4.  **Model Pipeline:**
    *   Create a `Pipeline` that includes:
        *   `StandardScaler()` for feature scaling.
        *   `LogisticRegression()` with the following parameters:
            *   `penalty='l2'`
            *   `C=1.0`
            *   `solver='liblinear'` (or another suitable solver like 'lbfgs' if preferred, ensure convergence)
            *   `class_weight='balanced'` (This is a key change to address class imbalance, as suggested by the paper).
            *   `max_iter=1000` (or more if needed for convergence)
            *   `random_state=42`

5.  **Model Training:**
    *   Train the pipeline on the **training set** (`X_train`, `y_train`).

6.  **Model Evaluation:**
    *   **On the Validation Set (`X_val`, `y_val`):**
        *   Make predictions.
        *   Calculate and print: Accuracy, Precision, Recall, F1-score, ROC AUC, and Precision-Recall AUC (AUPRC).
        *   Print the classification report and confusion matrix.
    *   **On the Test Set (`X_test`, `y_test`):**
        *   Make predictions.
        *   Calculate and print: Accuracy, Precision, Recall, F1-score, ROC AUC, and Precision-Recall AUC (AUPRC).
        *   Print the classification report and confusion matrix.
    *   Also, print the accuracy of the model on the training set (`X_train`, `y_train`) to monitor for overfitting.

7.  **Output:**
    *   The script should print all evaluation metrics clearly labeled for training, validation, and test sets.
    *   Save the Python script as `/procedure/pnet_prostate_paper/scripts/pnet_model_v2_sklearn_enhanced.py`.

**Success Criteria:**
- The script runs without errors.
- Data is correctly split into 80% train, 10% validation, 10% test, stratified by class.
- The Logistic Regression model with `class_weight='balanced'` is trained successfully.
- Evaluation metrics (Accuracy, Precision, Recall, F1, AUC, AUPRC) are reported for train, validation, and test sets.
- The script is saved to the specified path.

**Deliverables:**
1.  The Python script: `/procedure/pnet_prostate_paper/scripts/pnet_model_v2_sklearn_enhanced.py`.
2.  Console output showing the shapes of the data splits and all evaluation metrics for the training, validation, and test sets.

**Key Libraries to Use:**
- `pandas` for data manipulation.
- `scikit-learn` for: 
    - `train_test_split`
    - `StandardScaler`
    - `LogisticRegression`
    - `Pipeline`
    - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `average_precision_score` (for AUPRC), `classification_report`, `confusion_matrix`.

**Feedback Requirements:**
- Confirm that the data splitting (80/10/10 stratified) was implemented correctly by showing the shapes of `y_train`, `y_val`, and `y_test` and their class distributions (e.g., `value_counts(normalize=True)`).
- Clearly present all evaluation metrics for the training, validation, and test sets.
- Confirm the script was saved to `/procedure/pnet_prostate_paper/scripts/pnet_model_v2_sklearn_enhanced.py`.

```
