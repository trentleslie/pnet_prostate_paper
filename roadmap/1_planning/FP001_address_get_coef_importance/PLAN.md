# Implementation Plan: Address nn.Model.get_coef_importance

## 1. Phases & Tasks

**Phase 1: Investigation & Definition (Est: 1-2 hours)**

*   **Task 1.1:** Research the original behavior and output format of `get_coef_importance` and `self.coef_`. Check old versions of the code if available, or infer from how `self.coef_` was used.
*   **Task 1.2:** Clearly define what 'coefficient importance' means in the context of this model and what `self.coef_` should represent (e.g., `dL/da_l`, `dL/dw_l`, aggregated scores per feature).

**Phase 2: Design & Prototyping (Est: 2-3 hours)**

*   **Task 2.1:** Based on Task 1.2, determine if `get_activation_gradients` from `coef_weights_utils.py` provides a suitable basis.
*   **Task 2.2:** If not, or if transformation is needed, design the logic for a new helper function in `coef_weights_utils.py` to compute the desired importances.
*   **Task 2.3:** Prototype the calculation on a sample trained model if possible.

**Phase 3: Implementation (Est: 2-4 hours)**

*   **Task 3.1:** Implement the new helper function in `coef_weights_utils.py` (if designed in Task 2.2).
*   **Task 3.2:** Modify `nn.Model.get_coef_importance` in `/procedure/pnet_prostate_paper/model/nn.py` to use the new utility/logic and populate `self.coef_`.

**Phase 4: Testing & Validation (Est: 2-3 hours)**

*   **Task 4.1:** Write unit tests for the `get_coef_importance` method. This will require a way to instantiate a minimal trainable model and mock/provide necessary inputs.
*   **Task 4.2:** Manually inspect the `self.coef_` values on a sample trained model to ensure they are reasonable.

## 2. Dependencies

*   Clear definition of 'coefficient importance' for this model.
*   Access to a runnable/trainable version of the `nn.Model` to test the implementation.
*   Understanding of how `get_activation_gradients` and `resolve_gradient_function` work.

## 3. Milestones

*   **M1:** Definition of `self.coef_` structure and calculation method finalized.
*   **M2:** Helper function in `coef_weights_utils.py` (if any) implemented and unit-tested.
*   **M3:** `nn.Model.get_coef_importance` successfully populates `self.coef_` with expected values.
*   **M4:** All unit tests for `get_coef_importance` pass.
