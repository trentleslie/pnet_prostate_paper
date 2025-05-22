# Test Configurations for P-NET Prostate Paper Project

This directory houses configurations specifically designed for testing and development of the P-NET prostate cancer model codebase, particularly during and after its migration to Python 3.11 and TensorFlow 2.x.

## Purpose

The primary goal of these test configurations is to provide lightweight, reproducible setups to:

*   Test individual components of the model pipeline (e.g., data loading, model construction, custom callbacks) without requiring the full, original dataset or extensive computational resources.
*   Facilitate debugging and verification of refactored code.
*   Enable continuous integration and automated testing in the future.

## Directory Structure

*   `mock_params/`:
    *   Contains mock `*_params.yml` files.
    *   These YAML files define model parameters (`model_params`) and data parameters (`data_params`) necessary for initializing and running models via `utils.loading_utils.DataModelLoader` and `model.model_factory.construct_model`.
    *   A `template_params.yml` is provided as a comprehensive starting point, based on an example `_params.yml` found in the original project's logs. This template details the expected structure and common parameters.
    *   Specific mock files (e.g., `mock_basic_nn_params.yml`, `mock_nn_with_gradient_importance_params.yml`) are derived from this template for particular testing scenarios.

*   (Potentially in the future) `minimal_test_data/`:
    *   This directory might be added to house a truly minimal, self-contained dataset (e.g., a few samples, a subset of genes) for rapid testing where even pointing to the full dataset (in `data/_database/`) is too slow or cumbersome.

## Using Mock Parameter Files

1.  **Understand the Schema:** Refer to `mock_params/template_params.yml` and the project's data loading (`ProstateDataPaper`) and model construction (`build_pnet2`, `nn.Model`) code to understand the meaning of each parameter.

2.  **Select or Create a Mock File:** Choose an existing mock `_params.yml` from `mock_params/` that suits your testing needs, or create a new one by copying and modifying the template or an existing mock file.

3.  **Data Paths:**
    *   The `data_params` section in the YAML files specifies paths to data files (e.g., `selected_genes`, paths for mutation, CNV, expression data that `ProstateDataPaper` will load).
    *   For initial testing, these paths might point to the full-sized data files located in `/procedure/pnet_prostate_paper/data/_database/` (unzipped from `_database.zip`). Ensure these files are accessible.
    *   For more lightweight testing, paths might eventually point to a minimal dataset in `minimal_test_data/` (once created).

4.  **Running Tests/Scripts:**
    *   When running test scripts or parts of the main pipeline, you will typically need to provide the path to your chosen mock `_params.yml` file. The `DataModelLoader` class is usually the entry point for loading these configurations.
    *   For example, if a script expects a path to a directory containing `_params.yml` (as the original `one_split.py` did, looking in `_logs/.../experiment_name/`), you might need to structure your `test_configs/` or a temporary directory accordingly, or modify scripts to accept a direct path to the `_params.yml` file.

## Contribution

When adding new mock parameter files or test data:

*   Ensure they are well-documented (either within the file or in this README).
*   Keep them as minimal as possible while still serving their testing purpose.
*   If adding new data subsets, clearly explain their source and how they were derived.
