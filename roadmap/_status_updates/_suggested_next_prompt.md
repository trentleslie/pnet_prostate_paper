# Suggested Next Steps for P-NET Project

## 1. Context Brief
We have successfully debugged and refactored the SKCM tumor purity prediction scripts (`/procedure/pnet_prostate_paper/notebooks/SKCM_purity_tf2.py` and `/procedure/pnet_prostate_paper/scripts/run_skcm_purity_tf2.py`), which are now functional using synthetic data. The "P-NET Full Training Pipeline Integration Testing" feature is fully planned but awaits your input on several strategic questions before implementation can begin.

## 2. Initial Steps
1.  Please review the overall project context, guidelines, and workflow conventions documented in `/procedure/pnet_prostate_paper/CLAUDE.md`.
2.  Please review the latest status update summarizing our recent progress and current project state: `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-06-05-skcm-script-debug-and-project-status.md`.

## 3. Work Priorities
Based on the current project status, please consider the following priorities for your next actions:

1.  **Provide Answers to Strategic Questions for "P-NET Full Training Pipeline Integration Testing":**
    *   Your input is crucial to move this feature from "Planning" to "In Progress." The questions are detailed in the status update (Section 5) and were initially posed after the planning documents in `/procedure/pnet_prostate_paper/roadmap/1_planning/pnet_full_training_integration_testing/` were prepared.
    *   These questions cover: Priority/Timeline, Minimal Test Data Strategy, Performance Acceptance Criteria, `_params.yml` Replication Strategy, BCR Data Availability, and Baseline Model Comparisons.

2.  **Decide on Next Steps for SKCM Purity Scripts:**
    *   The scripts are functional with synthetic data.
    *   **Option A:** If you wish to run them with real SKCM data, please ensure the necessary data files (RNA-seq, CNA, purity) are available and correctly placed in `/procedure/pnet_prostate_paper/data/_database/skcm_tcga_pan_can_atlas_2018/`. Let Cascade know if you need assistance with verifying paths or adapting the scripts if filenames differ.
    *   **Option B:** If validation with synthetic data is sufficient for now, we can consider this task complete and focus on other priorities.

3.  **Address `_params.yml` Handling (`FP002_handle_missing_params_yml.md`):**
    *   This is a recurring critical item for full paper replication. If the integration testing is prioritized, a concrete plan for `_params.yml` will be needed soon.

## 4. Key References
*   **Latest Status Update:** `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-06-05-skcm-script-debug-and-project-status.md`
*   **SKCM Purity Scripts:**
    *   `/procedure/pnet_prostate_paper/notebooks/SKCM_purity_tf2.py`
    *   `/procedure/pnet_prostate_paper/scripts/run_skcm_purity_tf2.py`
*   **Integration Testing Planning Docs:** `/procedure/pnet_prostate_paper/roadmap/1_planning/pnet_full_training_integration_testing/`
*   **Prompt for SKCM Debugging (example of detailed prompt):** `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-05-074500-debug-skcm-purity-script.md`

## 5. Workflow Integration
For any chosen next step, remember that Cascade can assist in:
*   Analyzing requirements.
*   Generating detailed, actionable prompts for a Claude code instance if implementation or further investigation is needed.
*   Processing feedback from Claude and iterating on solutions.

Please indicate how you'd like to proceed.