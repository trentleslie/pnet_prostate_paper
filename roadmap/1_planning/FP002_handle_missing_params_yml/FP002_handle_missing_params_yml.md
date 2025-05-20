# Future Plan: Strategy for Missing _params.yml Files / Test Data Generation

**Date Identified:** 2025-05-20

**Source:** Status Update `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-05-20-tf2-migration-status.md`

**Problem:**
The `_logs/` directory and its contents, specifically the `*_params.yml` model configuration files, are missing from the current workspace. These files are critical for:
*   Loading and instantiating models with specific parameters.
*   Testing the `GradientCheckpoint` callback, which relies on `feature_importance` and `feature_names` settings within these YAML files.
*   Running various downstream analysis scripts that expect these configurations.

**Goal:**
Enable robust testing and full functionality of the model loading, training, and analysis pipelines by addressing the absence of these parameter files.

**Initial Thoughts/Requirements:**
1.  **Investigation:**
    *   Determine if the original `/procedure/pnet_prostate_paper/_logs/` directory and its `*_params.yml` files can be recovered (e.g., from a different branch, a backup, or another team member).
2.  **If Unrecoverable - Strategy Development:**
    *   **Mock/Template Parameter Files:** Define a schema and create a set of minimal, representative `*_params.yml` files. These should cover common scenarios and parameter variations needed for testing.
    *   **Minimal Test Model Setup:** Consider creating a simplified, self-contained model definition and corresponding dummy data that can be used specifically for testing callbacks like `GradientCheckpoint` without needing the full dataset or complex original parameter files.
    *   This strategy should allow for verification of model loading and callback functionality in the TF2.x environment.

**Next Step (as per HOW_TO_UPDATE_ROADMAP_STAGES.md):**
This item is ready for planning. Execute `roadmap/1_planning/STAGE_GATE_PROMPT_PLAN.md` for this feature.
