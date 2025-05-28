# Prompt: Implement `ignore_missing_histology` in `build_pnet` and Data Handling

**Date:** 2025-05-27
**Project:** P-NET TensorFlow 2.x Migration
**Source Prompt:** {{PROJECT_ROOT}}/roadmap/_active_prompts/2025-05-27-173800-impl-ign-hist-buildpnet.md
**Managed by:** Cascade (Project Manager AI)

## 1. Task Overview

The goal of this task is to implement the logic for the `ignore_missing_histology` parameter within the `build_pnet` function (and its related data handling components). Currently, no histology data is used by the model. This implementation should ensure that when `ignore_missing_histology` is `True` (the critical current use case), the model robustly builds and operates using only the existing genomic data types (mutations, CNAs, etc.). The implementation should also structure the code to cleanly allow for the *future, optional* integration of a distinct histology data pathway if `ignore_missing_histology` were `False` and such a pathway were developed.

This task is based on "Strategy 1 (Input Data Filtering / Conditional Pathway Bypass)" from the feedback file `{{PROJECT_ROOT}}/roadmap/_active_prompts/feedback/2025-05-24-073425-feedback-strat-ign-hist-buildpnet.md`.

## 2. Background and Context

*   The `ignore_missing_histology` boolean parameter is passed from YAML configurations through `model/model_factory.py` and `model/nn.py` to the `build_fn` (e.g., `build_pnet`).
*   The original P-NET paper (`{{PROJECT_ROOT}}/paper.txt`) indicates the model was primarily built using somatic mutation and copy number alteration data. Histology data was not an input in the published model.
*   The USER confirms no histology data is currently used, and the ability to explicitly bypass any histology-related processing is critical.
*   The primary model builder is `build_pnet` in `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`.
*   Data loading is handled by the `Data` class in `/procedure/pnet_prostate_paper/data/data_access.py`, configured by `data_params`.

## 3. Specific Tasks & Deliverables

1.  **Review `build_pnet` and `get_pnet`:**
    *   In `/procedure/pnet_prostate_paper/model/builders/prostate_models.py` (for `build_pnet`) and `/procedure/pnet_prostate_paper/model/builders_utils.py` (for `get_pnet`).
    *   Identify how input features are defined and processed.
    *   Ensure that the existing logic for genomic data (mutations, CNAs) remains the default operational path.

2.  **Adapt `build_pnet` (and `get_pnet` if necessary):**
    *   Modify `build_pnet` to explicitly use the `ignore_missing_histology` parameter.
    *   **If `ignore_missing_histology` is `True`:** The model should build and operate *exactly as it does now*, using only the available genomic data. No changes to the core architecture for this case are expected beyond ensuring the parameter is acknowledged.
    *   **If `ignore_missing_histology` is `False`:**
        *   For now, since no histology pathway exists, the behavior should be identical to when it's `True`.
        *   However, structure the code (e.g., with clear conditional blocks or points for future extension) where a histology-specific data input and processing pathway *could be added later*. This might involve placeholder comments or an empty conditional branch.
    *   The goal is to make the current genomic-only model the explicit behavior when the flag is true, and to prepare for future extension.

3.  **Review `Data` Class and `data_params` Usage:**
    *   Examine `/procedure/pnet_prostate_paper/data/data_access.py`.
    *   Ensure that the `Data` class, when initialized via `data_params` in `build_pnet`, correctly loads only the necessary genomic data and does not error if (hypothetical) histology-specific parameters or feature requests are absent, especially when `ignore_missing_histology` implies they shouldn't be there.
    *   Consider if `data_params` needs a new (optional) key that `build_pnet` could set based on `ignore_missing_histology` to signal to the `Data` class what to load (e.g., `data_params['include_histology_features'] = False`). This aligns with Strategy 1's idea of modifying `data_params`.

4.  **Update Keras/TensorFlow Calls:**
    *   While refactoring, ensure all Keras API calls, regularizers, initializers, etc., within the modified sections of `build_pnet` and `get_pnet` are TF2.x compatible, consistent with the ongoing migration.

5.  **Docstrings and Comments:**
    *   Add clear docstrings and comments explaining how `ignore_missing_histology` affects model construction and data loading, and where future histology-specific code would integrate.

## 4. Expected Output

*   Modified Python files:
    *   `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`
    *   Potentially `/procedure/pnet_prostate_paper/model/builders_utils.py`
    *   Potentially `/procedure/pnet_prostate_paper/data/data_access.py`
*   These files should contain the implemented logic for `ignore_missing_histology` as described.

## 5. Environment and Execution

*   All Python code execution, script running, and tool usage (like `pytest`) **must** be performed within the project's Poetry environment. This can typically be achieved by prefixing commands with `poetry run` (e.g., `poetry run python your_script.py`) or by activating the shell environment using `poetry shell` before running commands.

## 6. Feedback Requirements

Upon completion of the implementation, or if you encounter significant blockers or have clarifying questions:

1.  Create a feedback file named `YYYY-MM-DD-HHMMSS-feedback-impl-ign-hist-buildpnet.md` (e.g., `2025-05-27-180000-feedback-impl-ign-hist-buildpnet.md`, using UTC for timestamp when feedback is generated) in the `{{PROJECT_ROOT}}/roadmap/_active_prompts/feedback/` directory.
2.  The feedback file should contain:
    *   A reference to this source prompt: `{{PROJECT_ROOT}}/roadmap/_active_prompts/2025-05-27-173800-impl-ign-hist-buildpnet.md`.
    *   A summary of code changes made in each file.
    *   An explanation of how the `ignore_missing_histology` flag now controls behavior.
    *   Confirmation that TF2.x compatibility was maintained/updated in modified sections.
    *   Any issues encountered, assumptions made, or questions for the Project Manager (Cascade).
