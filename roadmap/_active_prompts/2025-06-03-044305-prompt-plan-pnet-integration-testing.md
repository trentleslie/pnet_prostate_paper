# Prompt for Claude Code Instance: Plan P-NET Full Training Integration Testing

**Task:** Execute the planning stage for the "P-NET Full Training Pipeline Integration Testing" feature.

**This task is defined by the prompt:** `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-03-044305-prompt-plan-pnet-integration-testing.md`

## Instructions:

You are a Claude code instance. Your task is to process a feature idea through the project's planning stage.

1.  **Input Stage Gate Prompt:**
    *   The primary instructions for this planning stage are located in the following file. You MUST read and follow the instructions within this file:
        *   `/procedure/pnet_prostate_paper/roadmap/1_planning/STAGE_GATE_PROMPT_PLAN.md`

2.  **Input Feature Idea File:**
    *   The feature idea to be planned is described in:
        *   `/procedure/pnet_prostate_paper/roadmap/0_backlog/pnet_full_training_integration_testing.md`

3.  **Execution:**
    *   Follow the instructions in `STAGE_GATE_PROMPT_PLAN.md`, using the content of `pnet_full_training_integration_testing.md` as the input "idea file".
    *   This will involve creating a new feature subfolder within `/procedure/pnet_prostate_paper/roadmap/1_planning/`. The suggested name for this subfolder, based on the idea file, would be `pnet_full_training_integration_testing`.
    *   Inside this new feature subfolder, you will generate the following planning documents:
        *   `README.md`
        *   `spec.md`
        *   `design.md`

4.  **Output - Feedback File:**
    *   Upon completion of the planning task (or if you encounter an unrecoverable error), you MUST create a detailed Markdown feedback file.
    *   **Feedback File Location:** `/procedure/pnet_prostate_paper/roadmap/_active_prompts/feedback/`
    *   **Feedback File Naming Convention:** Use the UTC timestamp of when you complete this task. The format should be `YYYY-MM-DD-HHMMSS-feedback-plan-pnet-integration-testing.md`. For example, if you finish on June 3rd, 2025, at 05:00:00 UTC, the filename would be `2025-06-03-050000-feedback-plan-pnet-integration-testing.md`.
    *   **Feedback File Content:**
        *   A summary of the actions you took.
        *   The full path to the new feature folder you created (e.g., `/procedure/pnet_prostate_paper/roadmap/1_planning/pnet_full_training_integration_testing/`).
        *   Confirmation that `README.md`, `spec.md`, and `design.md` were created within that folder.
        *   Any challenges encountered or assumptions made.
        *   Any questions you have for the Project Manager (Cascade) or the USER.
        *   A statement referencing this source prompt: "This task was executed based on the prompt: `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-03-044305-prompt-plan-pnet-integration-testing.md`".

**Important:**
*   Ensure all file paths you use or reference are absolute paths.
*   Adhere strictly to the instructions in `/procedure/pnet_prostate_paper/roadmap/1_planning/STAGE_GATE_PROMPT_PLAN.md`.
