# P-NET Model Development and Comparison Report

**Date:** June 05, 2025

## 1. Overview

This report summarizes the development and evaluation of predictive models for prostate cancer (primary vs. metastatic) based on somatic mutation and copy number alteration (CNA) data. The primary goal is to work towards validating and understanding the P-NET model presented by Elmarakeby et al. (Nature, 2021). This report details the data processing workflow, the scikit-learn based models developed, and a comparison of our latest model's performance against the published P-NET results.

## 2. Workflow and Scripts Used

The project involves several key stages, each supported by specific scripts and libraries:

### 2.1. Data Loading and Preprocessing

-   **Script:** `/procedure/pnet_prostate_paper/scripts/load_and_preprocess_pnet_data.py`
-   **Objective:** To load raw somatic mutation, CNA, and clinical response data, clean it, align samples and genes across datasets, and prepare it for model training.
-   **Key Libraries/Modules:** `pandas`, `numpy`.
-   **Input Data Files (from `/procedure/pnet_prostate_paper/data/_database/prostate/processed/`):
    *   Somatic Mutations: `P1000_final_analysis_set_cross_important_only_plus_hotspots.csv`
    *   Copy Number Alterations (CNA): `P1000_data_CNA_paper.csv`
    *   Clinical Response: `response_paper.csv`
-   **Processing Summary:**
    *   Loaded and indexed data by sample ID.
    *   Handled missing values in mutation data (filled with 0.0).
    *   Identified common samples and genes across all three datasets.
    *   Resulted in a final aligned dataset of **1,012 samples** and **9,205 genes** (combined mutation and CNA features, leading to 18,410 features in total for modeling).
-   **Output Data (passed to modeling scripts, typically as CSVs or DataFrames):
    *   `mutation_df_processed`
    *   `cna_df_processed`
    *   `response_df_processed`

### 2.2. Baseline Model (P-NET Model v1)

-   **Script:** `/procedure/pnet_prostate_paper/scripts/pnet_model_v1.py`
-   **Objective:** To establish an initial baseline performance using a standard scikit-learn classifier.
-   **Model Type:** Logistic Regression with L2 regularization.
-   **Key Libraries/Modules:** `pandas`, `scikit-learn` (specifically `StandardScaler`, `LogisticRegression`, `train_test_split`, and various metrics functions from `sklearn.metrics`).
-   **Key Settings & Workflow:**
    *   Combined mutation and CNA data into a single feature matrix.
    *   Data Split: 80% training, 20% testing (stratified).
    *   Feature Scaling: `StandardScaler`.
    *   Model Parameters: `C=1.0`, `solver='lbfgs'`, default class weights, `max_iter=1000`, `random_state=42`.
-   **Key Test Performance (as per Memory `a084f154`):
    *   Accuracy: 83.25%
    *   Precision (Metastatic): 86.67%
    *   Recall (Metastatic): 58.21%
    *   F1-score (Metastatic): 69.64%
    *   AUC-ROC: 90.09%
    *   Noted 100% training accuracy, indicating overfitting.

### 2.3. Enhanced Scikit-learn Model (P-NET Model v2)

-   **Script:** `/procedure/pnet_prostate_paper/scripts/pnet_model_v2_sklearn_enhanced.py`
-   **Objective:** To improve upon the baseline by incorporating methodologies from the P-NET paper, such as a dedicated validation set and class imbalance handling.
-   **Model Type:** Logistic Regression with L2 regularization.
-   **Key Libraries/Modules:** `pandas`, `scikit-learn` (similar to v1).
-   **Key Settings & Workflow (based on feedback file `2025-06-05-084159-feedback-pnet-model-v2-sklearn-enhanced.md`):
    *   Combined mutation and CNA data.
    *   Data Split: 80% training, 10% validation, 10% testing (all stratified).
    *   Feature Scaling: `StandardScaler`.
    *   Model Parameters: `C=1.0`, `solver='liblinear'`, **`class_weight='balanced'`**, `max_iter=1000`, `random_state=42`.

## 3. Performance Comparison: `pnet_model_v2` vs. P-NET Paper

The following table compares the test set performance of our enhanced Logistic Regression model (`pnet_model_v2`) with the results reported for the original P-NET model in the Elmarakeby et al. (2021) paper.

| Metric              | Our `pnet_model_v2` (Logistic Regression) | P-NET Paper (Elmarakeby et al., 2021) | Notes                                                                 |
| :------------------ | :---------------------------------------- | :------------------------------------ | :-------------------------------------------------------------------- |
| **AUC-ROC**         | **0.9403 (94.03%)**                       | 0.93 (93%)                            | Our model slightly higher.                                            |
| **AUPRC**           | **0.8863 (88.63%)**                       | 0.88 (88%)                            | Our model slightly higher.                                            |
| **Accuracy**        | **0.8824 (88.24%)**                       | 0.83 (83%)                            | Our model notably higher.                                             |
| Recall (Metastatic) | 0.8235 (82.35%)                           | Not directly reported as headline     | Our model shows strong recall for the minority class due to balancing. |

**Discussion of Comparison:**

Our `pnet_model_v2`, a Logistic Regression model enhanced with stratified 80/10/10 splits and balanced class weights, demonstrates surprisingly strong performance. Its key metrics (AUC-ROC, AUPRC, Accuracy) on the test set are comparable to, or even slightly exceed, those reported for the significantly more complex, custom-built P-NET deep learning model in the original publication.

This suggests that for this specific dataset and classification task, a well-tuned, simpler linear model can be highly effective. The `class_weight='balanced'` parameter was particularly impactful, substantially improving the recall for the metastatic class (from 58.21% in our v1 model to 82.35% in v2), which is crucial for imbalanced datasets.

However, a critical observation for `pnet_model_v2` is its **100% accuracy on the training set**, indicating overfitting. While test set performance is strong, this overfitting needs to be addressed to ensure the model's robustness and generalizability. The original P-NET paper likely employed more sophisticated regularization techniques inherent to deep learning or specific training protocols (e.g., adaptive learning rates, early stopping based on a validation set) that might have led to a model with better generalization, even if headline test metrics are similar.

The primary advantage of the original P-NET model, beyond predictive accuracy, might lie in its biologically informed architecture, potentially offering deeper insights and interpretability through its pathway-based structure (e.g., using DeepLIFT as mentioned in the paper).

## 4. Key Findings

1.  **Effective Preprocessing:** The data loading and alignment process successfully created a usable dataset of 1,012 samples and 9,205 gene-level features (18,410 total input features for models).
2.  **Strong Baseline Performance:** An enhanced Logistic Regression model (`pnet_model_v2`) achieved excellent predictive performance (Test AUC-ROC: 0.9403, Test AUPRC: 0.8863, Test Accuracy: 88.24%).
3.  **Class Imbalance Handled:** Using `class_weight='balanced'` in Logistic Regression significantly improved recall for the underrepresented metastatic class.
4.  **Overfitting Identified:** The `pnet_model_v2` exhibits overfitting (100% training accuracy), which requires attention.
5.  **Simple Models Can Be Powerful:** For this dataset, a simpler, well-tuned model rivals the performance of the complex P-NET architecture reported in the paper, at least on headline metrics.

## 5. Next Steps / Recommendations

1.  **Address Overfitting in `pnet_model_v2`:**
    *   Tune the regularization parameter `C` in Logistic Regression (e.g., try values like 0.001, 0.01, 0.1) using the validation set to select the optimal value.
    *   Consider L1 (LASSO) regularization for implicit feature selection, which might also help with high dimensionality.
2.  **Investigate P-NET GitHub Repository:**
    *   Explore the original P-NET authors' code (`https://github.com/marakeby/pnet_prostate_paper`) to understand their exact implementation, data input formats, and training procedures.
    *   If feasible, attempt to run their P-NET model using our preprocessed data for a more direct comparison of the architectures.
3.  **Feature Selection/Reduction:**
    *   Given the high number of features (18,410) relative to samples (1,012), explore explicit feature selection techniques (e.g., mutual information, recursive feature elimination) or dimensionality reduction (e.g., PCA).
4.  **Explore Alternative Models (Post-Overfitting Mitigation):**
    *   Consider tree-based models like Random Forest or Gradient Boosting (XGBoost, LightGBM), which can handle high-dimensional data well and are often robust to overfitting with proper tuning.

This report provides a snapshot of the current progress. Further iterations focusing on model robustness and deeper comparison with the original P-NET architecture will yield more comprehensive insights.
