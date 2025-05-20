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
