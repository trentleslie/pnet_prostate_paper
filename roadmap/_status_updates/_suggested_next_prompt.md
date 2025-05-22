## Suggested Next Prompt for P-NET TF2.x Migration

**Date:** 2025-05-22

### 1. Context Brief:
We've successfully refactored core TF2.x components including custom Keras layers (`layers_custom.py`), `model/nn.py` utilities (like `get_layer_output`), and the model factory (`model/model_factory.py`) with integrated parameter handling (e.g., `ignore_missing_histology`). The immediate focus shifts to refactoring the primary model building functions and making them utilize new parameters like `ignore_missing_histology`.

### 2. Initial Steps:
1.  Review overall project context and long-term goals in `/procedure/pnet_prostate_paper/roadmap/CLAUDE.md`.
2.  Review the latest status update at `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-05-22-tf2-core-refactoring-progress.md` for detailed recent accomplishments and next steps.

### 3. Work Priorities:
1.  **Refactor Primary Model Building Functions:**
    *   Target `build_pnet` and `build_pnet2` in `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`.
    *   Target `get_pnet` in `/procedure/pnet_prostate_paper/model/builders_utils.py`.
    *   Update Keras API calls, regularizers, initializers to be TF2.x compatible.
    *   Ensure compatibility with refactored custom layers and the model factory logic.
    *   **Crucially, implement logic within one of these builders (e.g., `build_pnet`) to utilize the `ignore_missing_histology` parameter to conditionally alter the model structure or input handling.** This is a key step to make the new parameter functional.
2.  **Verify `GradientCheckpoint` `gradient_function`:**
    *   Thoroughly test the `get_activation_gradients` function (from `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`) with a loaded model and actual data in a TF2.x environment to ensure it works as expected for the `GradientCheckpoint` callback.
3.  **Continue Progress on Planned Items:**
    *   Address `nn.Model.get_coef_importance` as per `/procedure/pnet_prostate_paper/roadmap/1_planning/FP001_address_get_coef_importance/PLAN.md`.
    *   Advance `_params.yml` integration and validation as per `/procedure/pnet_prostate_paper/roadmap/1_planning/FP002_handle_missing_params_yml/PLAN.md`.

### 4. Key File References:
*   **Model Builders:**
    *   `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`
    *   `/procedure/pnet_prostate_paper/model/builders_utils.py`
*   **Core Model Files:**
    *   `/procedure/pnet_prostate_paper/model/nn.py`
    *   `/procedure/pnet_prostate_paper/model/model_factory.py`
    *   `/procedure/pnet_prostate_paper/model/layers_custom.py`
    *   `/procedure/pnet_prostate_paper/model/callbacks_custom.py`
    *   `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`
*   **Configuration & Data:**
    *   Example `_params.yml` files (from `_logs.zip` extraction, e.g., `/procedure/pnet_prostate_paper/_logs/p1000/pnet/onsplit_average_reg_10_tanh_large_testing/P-net_ALL_params.yml`)
*   **Roadmap & Planning:**
    *   `/procedure/pnet_prostate_paper/roadmap/technical_notes/pnet_refactor.md`
    *   `/procedure/pnet_prostate_paper/roadmap/2_inprogress/05_tensorflow_migration.md`
    *   `/procedure/pnet_prostate_paper/roadmap/1_planning/FP001_address_get_coef_importance/PLAN.md`
    *   `/procedure/pnet_prostate_paper/roadmap/1_planning/FP002_handle_missing_params_yml/PLAN.md`

### 5. Workflow Integration (Cascade & Claude):
*   **Cascade:** Can assist with direct code modifications for refactoring builder functions, implementing the `ignore_missing_histology` logic, and setting up test harnesses for `GradientCheckpoint` and model builders.
*   **Claude:** Can be consulted for:
    *   Strategic advice on how best to modify `build_pnet` (or other builders) to handle `ignore_missing_histology` (e.g., "Based on the P-NET architecture described in `/procedure/pnet_prostate_paper/paper.txt` and the existing `build_pnet` function in `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`, what are 2-3 robust ways to modify the input layers or pathway concatenation if `ignore_missing_histology` is true? Consider implications for layer shapes, data flow, and maintaining biological relevance.").
    *   Generating boilerplate for test functions for the refactored model builders, ensuring different configurations (including `ignore_missing_histology` true/false) are covered.
    *   Reviewing complex refactored code sections for adherence to TF2.x best practices and potential subtle bugs.