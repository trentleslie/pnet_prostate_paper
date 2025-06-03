# Prompt: Adapt PyTorch P-NET Testing Script to TensorFlow 2.x for Prostate Cancer Project and Convert to Notebook

## 1. Goal

The primary goal is to create a Python 3.11 script that replicates the core workflow of the provided PyTorch-based P-NET testing script (`/procedure/pnet_prostate_paper/notebooks/testing.py`), but adapted for the `pnet_prostate_paper` project. This means using TensorFlow 2.x, the project's existing data loading and model building utilities, and the minimal prostate dataset.

The new script should serve as a basic integration test and example of training a P-NET model end-to-end within our project's framework. Subsequently, this script will be converted into a Jupyter Notebook.

## 2. Input Script for Reference

The conceptual workflow should be derived from:
*   `/procedure/pnet_prostate_paper/notebooks/testing.py`

**Note:** This reference script is PyTorch-based. Your task is to translate its *process and intent* into our TensorFlow 2.x environment, not to perform a line-by-line port.

## 3. Key Adaptations Required

*   **Framework:** Convert all operations from PyTorch to TensorFlow 2.x / Keras.
*   **Data Source:**
    *   Utilize the project's minimal prostate dataset located at `/procedure/pnet_prostate_paper/test_data/minimal_prostate_set/`. This dataset includes RNA, CNA, Mutation data, and clinical information including `Gleason_Binary` as the target.
    *   The reference script uses generic `rna.csv`, `cna.csv`, etc. from a `../data/test_data/` directory. These should be replaced by our structured minimal dataset.
*   **Data Loading & Preprocessing:**
    *   Use the `data.data_access.Data` class (from `/procedure/pnet_prostate_paper/data/data_access.py`) for loading and preprocessing data.
    *   The reference script uses a custom `pnet_loader`. This should be entirely replaced by our `Data` class and TensorFlow's `tf.data.Dataset` API if necessary for batching.
*   **Model Definition:**
    *   Use the `build_pnet` function (from `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`) to construct the P-NET model.
    *   Model parameters (e.g., layer sizes, activation functions, dropout rates) should be configurable, ideally loaded from a YAML configuration file (see section 4.3).
*   **Training Loop:**
    *   Implement the training loop using TensorFlow 2.x. This can be done using `tf.keras.Model.fit()` or a custom training loop if more control is needed.
    *   The reference script uses `nn.BCEWithLogitsLoss`. The TensorFlow equivalent (`tf.keras.losses.BinaryCrossentropy(from_logits=True)`) should be used.
    *   Use a standard optimizer like Adam (`tf.keras.optimizers.Adam`).
*   **Evaluation:**
    *   Calculate metrics like AUC using `sklearn.metrics.roc_auc_score` or TensorFlow equivalents.
    *   Generate an ROC curve plot using `matplotlib.pyplot`.
*   **Gene Lists/Pathways:**
    *   The reference script uses `gene_list` and `canc_genes` for feature selection and a GMT file for pathway information in its `run_geneset` variant.
    *   Our `Data` class and `build_pnet` function handle feature selection based on pathway maps and input data. Rely on these mechanisms. The script should demonstrate training a P-NET model that inherently uses the pathway structure defined by our project's pathway maps.
*   **Output Script Location:** The new script should be created at `/procedure/pnet_prostate_paper/scripts/run_minimal_pnet_training_tf2.py`.
*   **Logging:** Implement logging using Python's standard `logging` module. Log key steps, configurations, and results.
*   **Configuration:** Utilize a YAML configuration file for data and model parameters.

## 4. Detailed Steps for Implementation

### 4.1. Script Setup
*   Create the new Python script: `/procedure/pnet_prostate_paper/scripts/run_minimal_pnet_training_tf2.py`.
*   Import necessary libraries: `tensorflow`, `numpy`, `pandas`, `sklearn.metrics`, `matplotlib.pyplot`, `logging`, `yaml`, and relevant modules from the `pnet_prostate_paper` project (e.g., `Data`, `build_pnet`, utility functions for loading params).
*   Set up basic logging (e.g., to console and/or a file).

### 4.2. Configuration Loading
*   Create a YAML configuration file (e.g., `/procedure/pnet_prostate_paper/config/minimal_training_params.yml`).
*   This YAML file should follow the project's convention (see Memory `1486ce8c-b896-4b3c-b624-1930fbbbf453`), having `data_params` and `model_params` sections.
    *   `data_params`: Include paths to the minimal dataset components, target column name (`Gleason_Binary`), information about pathway maps, train/test split ratio, etc.
    *   `model_params`: Include parameters for `build_pnet` such as layer dimensions, dropout rates, learning rate, epochs, batch size, `ignore_missing_histology` (default to `True` as per Memory `d6e1a6b4-cc4f-4338-ac57-13af9ae0fe7c`).
*   Load these parameters in the script.

### 4.3. Data Loading and Preprocessing
*   Instantiate the `Data` class using `data_params` from the configuration file.
*   Load the minimal prostate dataset. The `Data` class should handle the merging of RNA, CNA, mutation data, and selection of features based on pathway maps.
*   Split the data into training and testing sets. The `Data` object might have methods for this, or you can use `sklearn.model_selection.train_test_split` on the sample IDs before fetching the full data matrices. Ensure the split is stratified by the target variable if appropriate.
*   The `Data` class should provide `x_train, y_train, x_test, y_test` (and corresponding sample IDs if needed for later analysis). These should be TensorFlow tensors or NumPy arrays.

### 4.4. Model Definition
*   Use the `build_pnet` function to create the Keras model, passing in `model_params` from the configuration.
*   Compile the model with an Adam optimizer, `tf.keras.losses.BinaryCrossentropy(from_logits=True)`, and metrics like `['accuracy', tf.keras.metrics.AUC(name='auc')]`.

### 4.5. Model Training
*   Train the model using `model.fit()` with `x_train`, `y_train`.
*   Use `x_test`, `y_test` for validation data during fitting.
*   Incorporate parameters like epochs and batch size from `model_params`.
*   Consider adding callbacks like `tf.keras.callbacks.EarlyStopping` and `tf.keras.callbacks.ModelCheckpoint` (as per Memory `cb244996-02e1-4df9-a2ab-e7b2473811e8` for checkpointing). Save checkpoints to a designated directory (e.g., `/procedure/pnet_prostate_paper/checkpoints/minimal_pnet/`).

### 4.6. Model Evaluation
*   After training, evaluate the model on the test set (`x_test`, `y_test`) using `model.evaluate()`. Log the results.
*   Get probability predictions for the test set: `y_pred_proba = model.predict(x_test)`.
*   Calculate ROC AUC score using `sklearn.metrics.roc_auc_score(y_test, y_pred_proba)`.
*   Plot the ROC curve using `matplotlib.pyplot`, similar to the reference script. Save the plot to a file (e.g., `/procedure/pnet_prostate_paper/results/minimal_pnet_roc_curve.png`).

### 4.7. Script Execution
*   Ensure the script can be run from the command line, e.g., `python /procedure/pnet_prostate_paper/scripts/run_minimal_pnet_training_tf2.py --config /procedure/pnet_prostate_paper/config/minimal_training_params.yml`.
*   Log all important steps, configurations, and final evaluation metrics.

## 5. File Paths and Project Structure

*   **Project Root:** `/procedure/pnet_prostate_paper/`
*   **New Script:** `/procedure/pnet_prostate_paper/scripts/run_minimal_pnet_training_tf2.py`
*   **New Notebook:** `/procedure/pnet_prostate_paper/notebooks/run_minimal_pnet_training_tf2.ipynb`
*   **Config File:** `/procedure/pnet_prostate_paper/config/minimal_training_params.yml` (you will need to define its content)
*   **Minimal Dataset:** `/procedure/pnet_prostate_paper/test_data/minimal_prostate_set/`
*   **Data Access Module:** `/procedure/pnet_prostate_paper/data/data_access.py`
*   **Model Builder Module:** `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`
*   **Output Checkpoints:** `/procedure/pnet_prostate_paper/checkpoints/minimal_pnet/`
*   **Output Results/Plots:** `/procedure/pnet_prostate_paper/results/`

## 6. Deliverables

1.  The Python script: `/procedure/pnet_prostate_paper/scripts/run_minimal_pnet_training_tf2.py`.
2.  The YAML configuration file: `/procedure/pnet_prostate_paper/config/minimal_training_params.yml` with sensible default values for running on the minimal dataset.
3.  The Jupyter Notebook: `/procedure/pnet_prostate_paper/notebooks/run_minimal_pnet_training_tf2.ipynb`, converted from the Python script.
4.  A brief explanation of how to run the script and notebook, and any assumptions made.
5.  Instructions or a small wrapper script to perform the conversion from the Python script to the Jupyter Notebook using `jupytext`.

## 7. Important Considerations

*   **Error Handling:** Implement basic error handling (e.g., for file not found).
*   **Reproducibility:** Set random seeds for TensorFlow and NumPy for reproducible results.
*   **Clarity:** Ensure the code is well-commented and follows Python best practices.
*   **Focus:** The initial focus is on adapting the core training and evaluation workflow. The reference script's `Pnet.run_geneset` and `GenesetNetwork` inspection parts are secondary; a successful adaptation of the basic `Pnet.run` equivalent is the priority. Our `build_pnet` already incorporates pathway information, so a separate "geneset run" might not be directly analogous but rather a configuration of `build_pnet`.

## 8. Post-processing: Convert to Jupyter Notebook

After generating the Python script (`/procedure/pnet_prostate_paper/scripts/run_minimal_pnet_training_tf2.py`), it needs to be converted to a Jupyter Notebook.

**Instructions for Conversion:**

1.  **Ensure Jupytext is installed:** If not, it can be installed via pip:
    ```bash
    pip install jupytext
    ```
2.  **Convert the script to a notebook:** Use the following command in the terminal, from the project root directory (`/procedure/pnet_prostate_paper/`):
    ```bash
    jupytext --to ipynb scripts/run_minimal_pnet_training_tf2.py -o notebooks/run_minimal_pnet_training_tf2.ipynb
    ```
    This will create `notebooks/run_minimal_pnet_training_tf2.ipynb`.

3.  **(Optional) Execute notebook in place:** If you need to verify the notebook runs correctly and embed outputs, you can use:
    ```bash
    jupyter nbconvert --to ipynb --inplace --execute --allow-errors notebooks/run_minimal_pnet_training_tf2.ipynb
    ```
    Ensure that the Python environment used by `jupyter nbconvert` has all necessary dependencies installed.

The primary deliverable for this conversion step is the `.ipynb` file. If you automate this conversion, please provide the commands used.

This task is defined by the prompt: `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-03-152441-adapt-testing-py-to-tf2-pnet.md`
