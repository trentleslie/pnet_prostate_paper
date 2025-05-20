# Design: Address nn.Model.get_coef_importance

## 1. Overview
The `nn.Model.get_coef_importance` method will be redesigned to utilize the TensorFlow 2.x compatible gradient and utility functions available in `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`. The core idea is to adapt or wrap these utilities to produce an output suitable for `self.coef_`.

## 2. Components Involved

*   **/procedure/pnet_prostate_paper/model/nn.py:**
    *   `Model.get_coef_importance()`: This method will be modified.
*   **/procedure/pnet_prostate_paper/model/coef_weights_utils.py:**
    *   `get_activation_gradients()`: May provide the raw gradient information.
    *   `resolve_gradient_function()`: Ensures correct gradient function resolution.
    *   A new helper function might be introduced here if complex transformation of `get_activation_gradients` output is needed.

## 3. Data Flow

1.  `nn.Model.get_coef_importance()` is called.
2.  It internally calls a (potentially new) utility function in `coef_weights_utils.py`, passing necessary model-specific information (e.g., layers, activations).
3.  The utility function calculates or retrieves the raw importance metrics (e.g., gradients of loss w.r.t. activations or weights).
4.  These raw metrics are processed/transformed into the final coefficient importance format.
5.  The result is assigned to `self.model.coef_` (or `self.coef_` directly within the `Model` class).

## 4. Detailed Design Considerations

*   **Determining Importance Metric:** The original logic of `get_coef_importance` needs to be understood or redefined. If it was based on `dL/da_l` (gradient of loss w.r.t. layer activations), then `get_activation_gradients` is a strong candidate. If it was based on weights or other metrics, the approach needs to be adapted.
*   **Output Format of `self.coef_`:** The expected structure of `self.coef_` must be determined. Is it a single array, a list of arrays per layer, or a dictionary? This will dictate the final transformation step.
*   **New Helper Function in `coef_weights_utils.py` (Potential):** If the logic to transform raw gradients (from `get_activation_gradients`) into the `self.coef_` format is complex or reusable, it should be encapsulated in a new function within `coef_weights_utils.py`.

## 5. Alternatives Considered

*   **Direct use of `get_activation_gradients`:** If `self.coef_` is intended to store raw activation gradients, this might be possible with minimal adaptation. However, 'coefficient importance' often implies a more processed value.
*   **Using external interpretability libraries (e.g., SHAP, LIME):** While powerful, this would be a larger deviation from the existing structure implied by the original `get_coef_importance` method and is likely out of scope for simply *restoring* this specific function.
