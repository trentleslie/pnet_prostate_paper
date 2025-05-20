# Feature: TensorFlow 1.x to TensorFlow 2.x (Keras) Migration

**Status:** In Progress

**Goal:** Update all TensorFlow and Keras related code from TF1.x APIs to TF2.x APIs, ensuring models define, train, and predict correctly.

**Tasks:**

1.  **Update Keras Imports:** (Partially Done)
    *   Changed `keras.*` imports to `tensorflow.keras.*` in `model/nn.py`.
    *   *Next Steps:* Ensure all files using Keras/TensorFlow have updated imports.
2.  **Update TensorFlow/Keras API Calls:** (Partially Done)
    *   `tf.random.set_random_seed` to `tf.random.set_seed` (e.g., in `train/run_me.py`). (Done)
    *   Model saving: Refactored `model.to_json()` & `model.save_weights()` to `model.save()` in `model/nn.py`, while attempting to keep JSON architecture for backward compatibility. (Done, but syntax error fix was problematic).
    *   Model loading: Reviewed `load_model` in `model/nn.py`; current `build_fn` pattern seems adaptable for TF2. (No changes made yet, further review needed).
    *   Keras backend functions (`K.function`, `K.learning_phase()`, `K.gradients`, `K.get_session()`):
        *   Refactored in `model/nn.py` (methods `get_layer_output`, `get_layer_outputs`). (Done)
        *   Refactored in `model/coef_weights_utils.py`: (Done for active code)
            *   `get_gradient`: Updated to use `tf.GradientTape`.
            *   `get_gradient_layer`: Updated to use `tf.GradientTape` with a temporary model.
            *   `get_weights_linear_model`: `K.function` removed.
            *   `get_skf_weights`: `K.function` removed.
            *   DeepExplain/SHAP related functions (`get_deep_explain_score_layer`, `get_shap_scores_layer`, etc.) using `K.function` or `K.get_session()` have been block-commented due to direct TF1 dependencies of `deepexplain` library. These will require library replacement or significant rework if functionality is to be retained.
        *   *Next Steps:* Verify no other active `K.*` backend calls remain in other critical path files.
3.  **Refactor Keras Model Definitions (`build_fn`):** (Todo)
    *   *Next Steps:* Inspect model definition functions (e.g., `pnet_multitask_graph`, other architectures specified in `params/*.py` files and called via `build_fn` in `model/nn.py`).
    *   Update layer definitions, model construction to use `tensorflow.keras.layers` and `tensorflow.keras.Model` (Functional API or Subclassing).
4.  **Update Custom Callbacks:** (Todo)
    *   *Next Steps:* Review `model/callbacks_custom.py` (`GradientCheckpoint`, `FixedEarlyStopping`) for compatibility with TF2 Keras callback API. Update as needed.
5.  **Review Training Loops:** (Todo)
    *   *Next Steps:* Examine `model.fit()` usage in `pipeline/one_split.py` and other pipeline classes. Ensure arguments and data formats are TF2 compatible.
6.  **Review Data Input Pipelines:** (Todo)
    *   *Next Steps:* Check how data (NumPy arrays, potentially `tf.data.Dataset` later) is fed to `model.fit()`. Ensure compatibility. Pandas `.as_matrix()` updated to `.to_numpy()` in `pipeline/one_split.py`.

**Files Recently Updated/Reviewed:**
-   `model/nn.py` (Imports, save/load, K.function refactoring)
-   `model/coef_weights_utils.py` (Refactored Keras backend functions, commented out DeepExplain/SHAP)
-   `train/run_me.py` (Seed setting)
-   `pipeline/one_split.py` (Pandas data conversion)

**Next Immediate Steps:**
-   Resolve syntax error in `model/nn.py` related to `save_model` logging (if still pending).
-   Begin refactoring a `build_fn` (e.g., `pnet_multitask_graph`) as a pilot for model definition updates (Task 3).
-   Address "Update Custom Callbacks" (Task 4), particularly verifying `GradientCheckpoint`'s `gradient_function` dependency (see `model/callbacks_custom.py`).
-   Continue review of "Review Training Loops" (Task 5) and "Review Data Input Pipelines" (Task 6).

**Blockers/Challenges:**
-   Complexity of P-NET architecture and custom layers/objectives might require careful refactoring.
-   Ensuring numerical stability and equivalent performance post-migration.
