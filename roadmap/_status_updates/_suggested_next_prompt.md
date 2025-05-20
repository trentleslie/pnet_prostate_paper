## Suggested Next Prompt for pnet_prostate_paper Project

**1. Context Brief:**
We have successfully updated the project roadmap, creating detailed planning documents for two key backlog items: addressing `nn.Model.get_coef_importance` (`FP001`) and handling missing `_params.yml` files (`FP002`). These items are now in the '1_planning' stage, and all roadmap changes have been committed and pushed. The immediate next coding priority is to implement the TensorFlow 2.x refactoring for the core model-building functions (`build_pnet2`, `get_pnet`) based on Claude's existing detailed plans.

**2. Initial Steps:**
*   Begin by reviewing the overall project context and AI collaboration guidelines documented in `/procedure/pnet_prostate_paper/roadmap/CLAUDE.md`.
*   Refresh your understanding of the latest progress by reviewing the most recent status update: `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-05-20-tf2-migration-status.md`.

**3. Work Priorities:**
*   **Primary Focus: Implement TF2.x Model Building Refactoring.**
    *   Based on Claude's technical notes in `/procedure/pnet_prostate_paper/roadmap/technical_notes/tensorflow_migration/`:
        *   Refactor `build_pnet2` in `/procedure/pnet_prostate_paper/prostate_models.py`.
        *   Refactor `get_pnet` in `/procedure/pnet_prostate_paper/model/builders_utils.py`.
        *   Refactor custom layers (`Diagonal`, `SparseTF`) as detailed in Claude's plan.
*   **Secondary (if time permits or primary is blocked): Advance Planning for `FP001` or `FP002`.**
    *   Review the `PLAN.md` within `/procedure/pnet_prostate_paper/roadmap/1_planning/FP001_address_get_coef_importance/` and begin executing its investigation/implementation steps.
    *   Alternatively, review the `PLAN.md` within `/procedure/pnet_prostate_paper/roadmap/1_planning/FP002_handle_missing_params_yml/` and begin its investigation/implementation steps.

**4. Key References:**
*   Status Update: `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-05-20-tf2-migration-status.md`
*   TF2 Migration Plan (by Claude): `/procedure/pnet_prostate_paper/roadmap/technical_notes/tensorflow_migration/`
*   `build_pnet2` target: `/procedure/pnet_prostate_paper/prostate_models.py`
*   `get_pnet` target: `/procedure/pnet_prostate_paper/model/builders_utils.py`
*   Planning for `FP001`: `/procedure/pnet_prostate_paper/roadmap/1_planning/FP001_address_get_coef_importance/`
*   Planning for `FP002`: `/procedure/pnet_prostate_paper/roadmap/1_planning/FP002_handle_missing_params_yml/`

**5. Workflow Integration with Claude:**
*   For the primary task of refactoring model-building functions, Claude has already provided a detailed plan. You can use Claude for:
    *   **Code Generation/Adaptation:** Provide Claude with specific code snippets from `build_pnet2` or `get_pnet` and ask for direct TF2.x equivalents, referencing its previous analysis.
        *   *Example Prompt for Claude:* "Based on your analysis in `/procedure/pnet_prostate_paper/roadmap/technical_notes/tensorflow_migration/`, please refactor the following Keras 1.x layer definition from `/procedure/pnet_prostate_paper/model/builders_utils.py` (lines X-Y) to be TF2.x compatible: [paste code snippet here]. Ensure all necessary imports from `tensorflow.keras` are included."
    *   **Debugging:** If you encounter errors during refactoring, provide Claude with the error message and the relevant code section for troubleshooting.
    *   **Unit Test Ideas:** Once a function is refactored, ask Claude to suggest unit test cases to verify its behavior.
*   For advancing `FP001` or `FP002`:
    *   Use Claude to help research specific technical questions outlined in the `PLAN.md` files or to generate boilerplate code for new scripts/modules defined in the design.