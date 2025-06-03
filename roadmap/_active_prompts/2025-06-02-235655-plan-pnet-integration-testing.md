# Prompt for Claude Code Instance: Plan P-NET Full Training Pipeline Integration Testing

**Date:** 2025-06-02
**Source Prompt File:** /procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-02-235655-plan-pnet-integration-testing.md

## 1. Task Overview
You are tasked with initiating the planning phase for the "P-NET Full Training Pipeline Integration Testing" feature. This involves processing a backlog item through the standard planning stage gate.

## 2. Instructions
Your primary set of instructions is located in the stage gate prompt:
*   **Stage Gate Prompt:** `/procedure/pnet_prostate_paper/roadmap/1_planning/STAGE_GATE_PROMPT_PLAN.md`

You must execute the instructions within that `STAGE_GATE_PROMPT_PLAN.md` file.

## 3. Input Files
*   **Backlog Item to Process:** `/procedure/pnet_prostate_paper/roadmap/0_backlog/09_pnet_training_integration_testing.md`
*   **Project Root (for context, if needed by STAGE_GATE_PROMPT_PLAN.md):** `/procedure/pnet_prostate_paper/`

## 4. Expected Outputs & Deliverables
As per `STAGE_GATE_PROMPT_PLAN.md`, you are expected to:
1.  Create a new feature-specific subfolder within `/procedure/pnet_prostate_paper/roadmap/1_planning/`. The subfolder should be named after the backlog item, likely `09_pnet_training_integration_testing`.
2.  Inside this new subfolder, generate the following planning documents, populating them based on the input backlog item and the instructions in `STAGE_GATE_PROMPT_PLAN.md`:
    *   `README.md`
    *   `spec.md`
    *   `design.md`
3.  Ensure all paths used and generated are absolute paths.

## 5. Feedback Requirements
Upon completion of this task (successfully or with errors), you MUST create a detailed Markdown feedback file.
*   **Feedback File Location:** `/procedure/pnet_prostate_paper/roadmap/_active_prompts/feedback/`
*   **Feedback File Naming Convention:** `YYYY-MM-DD-HHMMSS-feedback-plan-pnet-integration-testing.md` (use the UTC timestamp of when you complete this task).
*   **Feedback File Content:**
    *   Reference this source prompt: `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-02-235655-plan-pnet-integration-testing.md`.
    *   Detail all actions taken.
    *   List all files created or modified, with their absolute paths.
    *   Describe any issues encountered and how they were handled.
    *   If you have any questions or need clarifications for the next stage, include them.

## 6. Tool Usage
*   You will likely need tools for file writing (e.g., `Write` tool). The `STAGE_GATE_PROMPT_PLAN.md` will provide specifics on document creation.
*   Ensure all file operations use absolute paths.

Please proceed with processing the backlog item through the planning stage gate.
