# Future Plan: Address nn.Model.get_coef_importance

**Date Identified:** 2025-05-20

**Source:** Status Update `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-05-20-tf2-migration-status.md`

**Problem:**
The `nn.Model.get_coef_importance` method in `/procedure/pnet_prostate_paper/model/nn.py` relies on a global `get_coef_importance` function that was previously in `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`. This global function has been removed or significantly refactored as part of the TF2 migration, breaking the functionality of `nn.Model.get_coef_importance`.

**Goal:**
Restore the functionality of `nn.Model.get_coef_importance` so that the model can correctly calculate and store its `self.coef_` attribute. This is important for understanding feature/coefficient importances from the trained model.

**Initial Thoughts/Requirements:**
*   Investigate how the new `resolve_gradient_function` and `get_activation_gradients` (both in `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`) can be used or adapted to provide the necessary information for `self.coef_`.
*   The `self.coef_` attribute likely expects a specific format (e.g., a NumPy array or a list of arrays corresponding to layers/features).
*   Consider if a new wrapper function in `coef_weights_utils.py` is needed to recreate the specific output format previously expected by `nn.Model.get_coef_importance`.
*   Update `/procedure/pnet_prostate_paper/model/nn.py` to call the new or adapted mechanism.

**Next Step (as per HOW_TO_UPDATE_ROADMAP_STAGES.md):**
This item is ready for planning. Execute `roadmap/1_planning/STAGE_GATE_PROMPT_PLAN.md` for this feature.
