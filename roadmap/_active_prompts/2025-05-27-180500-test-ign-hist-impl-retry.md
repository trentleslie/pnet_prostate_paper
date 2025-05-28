# Prompt: Test Implementation of `ignore_missing_histology`

**Date:** 2025-05-27
**Project:** P-NET TensorFlow 2.x Migration
**Source Prompt:** {{PROJECT_ROOT}}/roadmap/_active_prompts/2025-05-27-180500-test-ign-hist-impl-retry.md
**Managed by:** Cascade (Project Manager AI)

## 1. Task Overview

The goal of this task is to thoroughly test the recent implementation of the `ignore_missing_histology` parameter in `build_pnet`, `build_pnet2` (in `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`), and the `Data` class (in `/procedure/pnet_prostate_paper/data/data_access.py`). The implementation details are outlined in the feedback file: `/procedure/pnet_prostate_paper/roadmap/_active_prompts/feedback/2025-05-27-174031-feedback-impl-ign-hist-buildpnet.md`.

Testing should verify:
*   No regressions in existing model building functionality.
*   Correct behavior when `ignore_missing_histology` is explicitly set to `True` or `False`.
*   Accurate logging output corresponding to the selected mode.
*   Continued TF2.x compatibility.

## 2. Background and Context

*   The `ignore_missing_histology` parameter was implemented to allow the P-NET model to robustly operate using only genomic data (current state) and to provide a clear extension point for future histology data integration.
*   When `True` (or by default), the model should build using only genomic data.
*   When `False`, the model should currently log a warning and fall back to genomic-only mode, as the histology pathway is not yet implemented.
*   The `Data` class now has an `include_histology_features` parameter, which is influenced by `ignore_missing_histology` via `data_params`.

## 3. Specific Testing Tasks & Deliverables

1.  **Identify Test Entry Points:**
    *   Determine suitable existing training scripts, model instantiation scripts, or unit tests that utilize `build_pnet` or `build_pnet2`. If no single script is ideal, you may need to create a minimal test script.
    *   Focus on tests that involve model creation and potentially a very short training run or data pass-through to ensure the architecture is sound.

2.  **Regression Testing (Default Behavior):**
    *   Run the identified test(s) using existing configurations where `ignore_missing_histology` would not be explicitly set (thus defaulting to `True` as per the implementation).
    *   **Expected Outcome:** The model should build and operate without errors, identically to its behavior before these changes. Verify that logs indicate "genomic data only" mode or similar, as implemented.

3.  **Explicit `ignore_missing_histology=True` Test:**
    *   Modify the test setup or a configuration file to explicitly pass `ignore_missing_histology=True` to `build_pnet`/`build_pnet2`.
    *   **Expected Outcome:** The model should build and operate without errors. Logs should clearly indicate "genomic data only" mode. Behavior should be identical to the default.

4.  **Explicit `ignore_missing_histology=False` Test:**
    *   Modify the test setup or a configuration file to explicitly pass `ignore_missing_histology=False` to `build_pnet`/`build_pnet2`.
    *   **Expected Outcome:**
        *   The model should still build and operate without errors (falling back to genomic-only mode).
        *   A warning should be logged indicating that histology features are requested but not implemented, and the model is proceeding with genomic data only.
        *   Logs should reflect the attempt to include histology but the fallback to genomic-only.

5.  **Verify `data_params` and `Data` Class Interaction:**
    *   If possible within the test setup, inspect or log the `data_params` being passed to the `Data` class initializer.
    *   **Expected Outcome:**
        *   When `ignore_missing_histology` is `True` (or default), `data_params` should contain `include_histology_features: False` (or equivalent based on the implementation).
        *   When `ignore_missing_histology` is `False`, `data_params` should reflect this intent (e.g. `include_histology_features: True`, which then triggers the warning in the `Data` class).

6.  **TF2.x Compatibility Check:**
    *   Ensure all tests run successfully in the TensorFlow 2.x environment.
    *   Confirm that any Keras/TF API calls made during model construction in the tests are compatible.

7.  **Test Script(s) and Configuration (Deliverable):**
    *   If new test scripts are created, provide them.
    *   If existing configurations are modified for testing, detail these modifications.

## 4. Expected Output

*   A feedback file detailing:
    *   Test procedures followed.
    *   Modifications made to existing scripts/configurations or new test scripts created.
    *   Confirmation of expected outcomes for each test case (or details of any deviations/failures).
    *   Snippets of relevant log outputs verifying correct mode selection and warnings.
    *   Confirmation of TF2.x compatibility.

## 5. Environment and Execution

*   All Python code execution, script running, and tool usage (like `pytest`) **must** be performed within the project's Poetry environment: `poetry run ...` or `poetry shell`.
*   The `claude` command, if needed by you for auxiliary tasks, requires `nvm use --delete-prefix v22.15.1 && claude ...`.

## 6. Feedback Requirements

Upon completion of testing, or if you encounter significant blockers or have clarifying questions:

1.  Create a feedback file named `YYYY-MM-DD-HHMMSS-feedback-test-ign-hist-impl.md` in the `{{PROJECT_ROOT}}/roadmap/_active_prompts/feedback/` directory.
2.  The feedback file should contain:
    *   A reference to this source prompt.
    *   A detailed summary of tests performed and their results (pass/fail, log snippets).
    *   Any test scripts created or modifications to existing files.
    *   Any issues encountered, assumptions made, or questions for the Project Manager (Cascade).
**IMPORTANT:** If you are unable to write the feedback file to the specified path for any reason, please print the COMPLETE intended content of the feedback file to standard output before terminating.
