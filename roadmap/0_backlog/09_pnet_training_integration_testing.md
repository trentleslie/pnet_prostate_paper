# Backlog Item: P-NET Full Training Pipeline Integration Testing

## 1. Overview
This task involves validating the P-NET models, which have been recently stabilized through comprehensive debugging of their building functions, in an end-to-end training scenario. The goal is to ensure that these models can be successfully integrated into a full training pipeline, including data loading, model forward and backward passes, and basic metric calculation.

## 2. Problem Statement
The P-NET model building components (`build_pnet`, `build_pnet2`, `get_pnet`, and custom layers like `SparseTFSimple`) have been unit-tested and debugged. However, their successful operation within a complete training loop has not yet been verified. Potential issues could arise from data incompatibilities, incorrect shape handling during training, or problems with gradient flow and optimization when the full model is assembled and run.

## 3. Key Requirements
*   **Training Script:** Develop or adapt an existing training script capable of:
    *   Loading or generating mock/simplified training, validation, and test datasets.
    *   Instantiating a P-NET model using the debugged builder functions (e.g., `get_pnet`).
    *   Performing a standard training loop (data iteration, forward pass, loss calculation, backward pass, optimizer step).
    *   Calculating and logging basic metrics (e.g., loss, accuracy if applicable).
*   **Data Handling:**
    *   Implement mechanisms to load or generate mock datasets suitable for training a P-NET model. This should include mock genomic data (mutations, CNAs) and pathway maps.
    *   Ensure data formats are compatible with the model's input layers.
*   **Model Instantiation:**
    *   Utilize the existing `get_pnet` (or `build_pnet`/`build_pnet2`) functions to construct the model.
    *   Configure the model with appropriate parameters for a test run (e.g., simplified architecture, small number of epochs).
*   **Basic Validation:**
    *   Confirm that the model can train for a few epochs without crashing.
    *   Verify that loss values are generated and (ideally) decrease.
    *   Check that gradients are flowing (no `None` gradients for trainable weights).

## 4. Success Criteria
*   A P-NET model, built using the debugged functions, successfully completes a short training run (e.g., 3-5 epochs) on mock or simplified data without runtime errors.
*   The training script logs loss values, and these values demonstrate sensible behavior (e.g., not NaN, ideally decreasing).
*   The setup allows for easy configuration of different P-NET architectures (as supported by `get_pnet`) for integration testing.
*   The integration test provides a foundational framework that can be expanded for more comprehensive training validation later.

## 5. Key References
*   Status Update: `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-05-28-pnet-test-suite-debugged.md`
*   Completed Debugging Summary: `/procedure/pnet_prostate_paper/roadmap/3_completed/pnet_model_test_debugging/summary.md`
*   Main Test File (for model builders): `/procedure/pnet_prostate_paper/model/testing/test_model_builders.py`
*   Key Model Building Utility: `/procedure/pnet_prostate_paper/model/builders/builders_utils.py` (contains `get_pnet`)
*   Custom Layers: `/procedure/pnet_prostate_paper/model/layers_custom_tf2.py`
