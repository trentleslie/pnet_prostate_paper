# Suggested Next Prompt for P-NET Project

## Context Brief

The detailed planning phase for "P-NET Full Training Pipeline Integration Testing" is now complete. The planning documents (`README.md`, `spec.md`, `design.md` located in `/procedure/pnet_prostate_paper/roadmap/1_planning/pnet_full_training_integration_testing/`) have been updated to align with the primary goal of replicating the Elmarakeby et al., Nature 2021 paper using the modernized Python 3.11/TensorFlow 2.x P-NET codebase. We are now awaiting your input on several key strategic questions to finalize this plan and move the feature into implementation.

## Initial Steps

1.  Review the overall project context and your role by reading `/procedure/pnet_prostate_paper/CLAUDE.md`.
2.  Review the latest status update document: `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-06-03-pnet-integration-test-planning-update.md` for a comprehensive overview of recent accomplishments, current project state, and detailed next steps.

## Work Priorities

1.  **Address Open Strategic Questions:** Provide answers to the questions listed in Section 5 ("Open Questions & Considerations") of the status update (`/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-06-03-pnet-integration-test-planning-update.md`). These are critical for finalizing the "P-NET Full Training Pipeline Integration Testing" plan. The questions cover:
    *   Priority & Timeline for moving the feature to "In Progress".
    *   Minimal Test Data strategy for Phase 1.
    *   Performance Acceptance Criteria against paper results.
    *   `_params.yml` Replication Strategy (related to roadmap item `FP002`).
    *   Biochemical Recurrence (BCR) Data Availability and fallback.
    *   Approach for Baseline Model Comparisons.
2.  **Approve "In Progress" Transition:** Based on your answers and satisfaction with the current plan, formally approve moving the "P-NET Full Training Pipeline Integration Testing" feature to the "In Progress" stage of the roadmap.
3.  **Initiate "In Progress" Stage Gate (If Approved):** If you approve the transition, instruct Cascade to generate and execute the "In Progress" stage gate prompt for the Claude code instance. This prompt will instruct Claude to create detailed task lists and implementation notes for the integration testing feature, based on the updated planning documents.
4.  **Plan `_params.yml` (`FP002`) Solution:** Discuss and define a concrete plan of action for addressing the `_params.yml` challenge (roadmap item `FP002_handle_missing_params_yml.md`). This is crucial for the later phases of integration testing that involve real data and specific experimental setups from the paper.

## References

*   Latest Status Update: `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-06-03-pnet-integration-test-planning-update.md`
*   Integration Testing Planning Documents:
    *   `/procedure/pnet_prostate_paper/roadmap/1_planning/pnet_full_training_integration_testing/README.md`
    *   `/procedure/pnet_prostate_paper/roadmap/1_planning/pnet_full_training_integration_testing/spec.md`
    *   `/procedure/pnet_prostate_paper/roadmap/1_planning/pnet_full_training_integration_testing/design.md`
*   Source Paper: `/procedure/pnet_prostate_paper/paper.txt`
*   Refactoring Technical Notes: `/procedure/pnet_prostate_paper/roadmap/technical_notes/pnet_refactor.md`

## Workflow Integration

Please provide your responses to the strategic questions first. Once these are clarified, you can direct Cascade to proceed with the roadmap stage transition for the integration testing feature. For the `_params.yml` task, outline your preferred approach or any initial thoughts to guide further planning and implementation.