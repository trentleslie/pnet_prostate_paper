# Development Status Update - 2025-05-20

## 1. Recent Accomplishments (In Recent Memory)
- **`model/model_utils.py` Refactoring:** Updated for Python 3.11 and TensorFlow 2.x compatibility. This included replacing `cPickle` with `pickle`, migrating Keras imports (e.g., `keras.models.Sequential` to `tensorflow.keras.models.Sequential`), and converting Python 2 `print` statements and `file()` built-ins to their Python 3 equivalents.
- **`model/nn.py` Layer Output Refactoring:** The methods `get_layer_output` and `get_layer_outputs` were successfully refactored to use `tensorflow.keras.models.Model` (aliased as `KerasModel`) for extracting intermediate layer activations, replacing the deprecated `K.function` and `K.learning_phase()` from TensorFlow 1.x.
- **`model/callbacks_custom.py` Initial TF2 Update:**
    - Updated `keras.callbacks.Callback` import to `tensorflow.keras.callbacks.Callback`.
    - Investigated and confirmed Python 2 `print` statements were already handled by previous `2to3` conversion.
    - The core callback structure appears largely compatible, but functional testing depends on deeper dependencies (like `gradient_function`).
- **`model/coef_weights_utils.py` Refactoring (In Progress):**
    - Updated Keras imports (e.g., `keras.backend` to `tensorflow.keras.backend`).
    - Corrected a typo: `get_gradeint` renamed to `get_gradient`.
    - Iteratively fixing Python 2 `print` statements to Python 3 `print()` functions. Several have been addressed, but one `SyntaxError` (ID: `212e8fb5-ab49-485a-acf3-39cd29958279`) remains on line 277.
    - Identified key TF1 patterns (`K.function`, `model.optimizer.get_gradients`, `K.get_session()`) that require substantial refactoring for TF2.

## 2. Current Project State
- **Overall:** The project is in an active phase of refactoring to ensure compatibility with Python 3.11 and TensorFlow 2.x. The primary focus is on migrating TensorFlow/Keras specific code.
- **Key Modules Status:**
    - `/procedure/pnet_prostate_paper/model/nn.py`: Core methods for layer output retrieval are updated. Model saving/loading mechanisms were addressed in earlier sessions. The `build_fn` part for model definitions is pending refactoring.
    - `/procedure/pnet_prostate_paper/model/model_utils.py`: Substantially updated for Python 3 and TF2. The function `get_coef_importance` is currently a focus due to its critical dependency on `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`.
    - `/procedure/pnet_prostate_paper/model/callbacks_custom.py`: Syntactic updates for TF2 are mostly complete. The functional compatibility of `GradientCheckpoint` heavily depends on the refactoring of its `gradient_function` (likely from `get_coef_importance`).
    - `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`: This is the current "hotspot" for active refactoring. Basic Python 3 syntax (imports, print statements) is being addressed. The more complex task of converting TF1-specific gradient and backend function calls to TF2 (e.g., using `tf.GradientTape`) is imminent.
- **Stability:**
    - General Python 2 to 3 syntax conversions (e.g., `print` statements, `cPickle` to `pickle`, `xrange` to `range`) handled by `2to3` or earlier manual edits should be largely stable across the codebase.
    - TensorFlow/Keras API migration is highly active and files undergoing this process are in a transient state.
- **Outstanding Critical Issues or Blockers:**
    - **`kaleido` Package Installation:** The installation of `kaleido` (version `0.2.1.post1`) for Plotly static image export remains deferred due to environment compatibility issues (Memory `c5400c15-0684-4c3e-974d-4ce16f88ad54`). This does not block core TF migration but will need addressing for full feature parity.
    - **TF2 Compatibility of `coef_weights_utils.py`:** The functions within this file are critical for `get_coef_importance` and, by extension, the `GradientCheckpoint` callback. Until this file is fully TF2-compatible, these higher-level components cannot be reliably used or tested.
    - **SyntaxError in `coef_weights_utils.py`:** A `SyntaxError: Missing parentheses in call to 'print'` (ID: `212e8fb5-ab49-485a-acf3-39cd29958279`) persists on line 277 of `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`.

## 3. Technical Context
- **Recent Architectural Decisions:**
    - **Layer Output Extraction:** Adopted the pattern of creating a temporary `tensorflow.keras.models.Model` to extract intermediate layer activations in `/procedure/pnet_prostate_paper/model/nn.py`, replacing `K.function`.
    - **Refactoring Order:** Prioritized fixing basic Python syntax and imports in `/procedure/pnet_prostate_paper/model/coef_weights_utils.py` before tackling its core TF1 logic, as this provides a cleaner base for more complex changes.
- **Key Data Structures, Algorithms, or Patterns (Targeting for TF2):**
    - **`tf.GradientTape`:** This will be the primary mechanism for replacing TF1-style gradient calculations that currently use `K.gradients` or `model.optimizer.get_gradients` in conjunction with `K.function`.
    - **Eager Execution:** All refactoring is done with TensorFlow 2.x's eager execution model in mind, eliminating the need for explicit session management (`K.get_session()`).
- **Important Learnings:**
    - The `2to3` tool is effective for many Python 2 to 3 conversions, but careful manual review and iterative fixes are often necessary, especially for `print` statements within complex code or those missed by automated tooling.
    - Migrating from TF1 Keras backend functions (`K.*`) to TF2 often involves a shift in approach, e.g., from symbolic graph construction and `K.function` to direct model calls and `tf.GradientTape`.
- **Specific Implementation Details to Remember:**
    - A typo `get_gradeint` was corrected to `get_gradient` in the definition and calls within `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`.

## 4. Next Steps
- **Immediate Tasks:**
    1.  **Resolve SyntaxError:** Fix the `SyntaxError: Missing parentheses in call to 'print'` (ID: `212e8fb5-ab49-485a-acf3-39cd29958279`) on line 277 of `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`.
    2.  **Continue `coef_weights_utils.py` Refactoring:**
        *   Replace all instances of `K.function` and associated logic (e.g., `model.optimizer.get_gradients`, manual input tensor assembly) with `tf.GradientTape` for gradient computations.
        *   Address `K.get_session()` usage in `get_deep_explain_scores` (likely by ensuring DeepExplain or an alternative works with eager execution or by reimplementing the required "grad*input" logic).
        *   Update the `keras.wrappers.scikit_learn.KerasClassifier` usage in `get_skf_weights` to its `tensorflow.keras` or `scikeras` equivalent.
- **Priorities for the Coming Week:**
    1.  Complete the full refactoring of `/procedure/pnet_prostate_paper/model/coef_weights_utils.py` for TensorFlow 2.x compatibility.
    2.  Once `coef_weights_utils.py` is stable, thoroughly test `/procedure/pnet_prostate_paper/model/model_utils.py::get_coef_importance`.
    3.  Test the `/procedure/pnet_prostate_paper/model/callbacks_custom.py::GradientCheckpoint` callback.
    4.  Begin refactoring Keras model definitions (functions referenced by `build_fn` in `/procedure/pnet_prostate_paper/model/nn.py`, often defined in `params/*.py` files) to use `tensorflow.keras.layers` and `tensorflow.keras.Model` Functional API or Subclassing.
- **Dependencies or Prerequisites:**
    - Testing `get_coef_importance` and `GradientCheckpoint` is dependent on a successfully refactored `coef_weights_utils.py`.
- **Potential Challenges or Considerations:**
    - Ensuring numerical equivalence or acceptable similarity of results from gradient/importance calculation functions in `coef_weights_utils.py` after refactoring from TF1 to TF2.
    - The complexity of some functions in `coef_weights_utils.py` might make the `tf.GradientTape` conversion non-trivial.

## 5. Open Questions & Considerations
- **DeepExplain in TF2:** How will the `deepexplain` library, particularly its reliance on `K.get_session()` in `get_deep_explain_scores`, be handled? Will it support TF2's eager execution, or is an alternative (like a focused SHAP implementation or manual "grad*input") necessary?
- **`GradientCheckpoint` Naming:** Is the name `GradientCheckpoint` in `/procedure/pnet_prostate_paper/model/callbacks_custom.py` still appropriate? It primarily saves gradient/coefficient history to CSV files rather than model checkpoints. (This is a minor point, but worth noting for clarity).
- **Testing Strategy for Refactored Utilities:** What is the best approach to test the refactored functions in `coef_weights_utils.py` for correctness, especially if direct numerical comparison with TF1 versions is difficult to set up? Could involve testing on simple, known models or comparing output distributions.
