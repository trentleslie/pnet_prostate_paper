# Prompt for Claude Code Instance: Progress P-NET Full Training Pipeline Integration Testing to In-Progress

**Date:** 2025-06-03
**Source Prompt File:** /procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-03-001825-progress-pnet-integration-testing.md

## 1. Task Overview
You are tasked with transitioning the "P-NET Full Training Pipeline Integration Testing" feature to the "in-progress" stage. This involves processing the existing planning documents through the standard in-progress stage gate.

## 2. Instructions
Your primary set of instructions is located in the stage gate prompt:
*   **Stage Gate Prompt:** `/procedure/pnet_prostate_paper/roadmap/2_inprogress/STAGE_GATE_PROMPT_PROG.md`

You must execute the instructions within that `STAGE_GATE_PROMPT_PROG.md` file.

## 3. Input Files & Folders
*   **Feature Planning Folder to Process:** `/procedure/pnet_prostate_paper/roadmap/1_planning/09_pnet_training_integration_testing/`
    *   This folder contains `README.md`, `spec.md`, and `design.md` which you must use as input.
*   **Project Root (for context):** `/procedure/pnet_prostate_paper/`
*   **Original Paper (for data source investigation):** `/procedure/pnet_prostate_paper/paper.txt`

## 4. Key Decisions & Context for Implementation Planning
When generating `task_list.md` and `implementation_notes.md` (as per `STAGE_GATE_PROMPT_PROG.md`), incorporate the following decisions and guidance:

*   **Data Strategy:**
    *   The primary approach will be to use **simplified real data**.
    *   **Action:** Investigate the original paper (`/procedure/pnet_prostate_paper/paper.txt`), particularly its "Data availability" section, Supplementary Tables (1-5), and any links to external data sources or repositories (e.g., the mentioned GitHub repository `https://github.com/marakeby/pnet_prostate_paper`).
    *   **Output:** In `implementation_notes.md`, outline a strategy to identify and obtain a small, representative subset of the original training/validation data. This subset should be manageable for initial integration testing (e.g., a few dozen samples). Detail how this subset will be created and stored.
*   **Test Framework Location:**
    *   A **new dedicated directory** should be created for these integration tests.
    *   **Output:** In `implementation_notes.md`, propose a suitable name and location for this new directory (e.g., `/procedure/pnet_prostate_paper/integration_tests/` or `/procedure/pnet_prostate_paper/pipeline/testing/`).
*   **Model Variant:**
    *   The initial implementation should focus on a **single, relatively simple P-NET variant**.
    *   **Output:** In `implementation_notes.md`, identify a suitable candidate P-NET configuration from the existing builders (e.g., a model with one or two sparse layers) and justify the choice.
*   **Resource Usage & Performance:**
    *   The integration test script should be designed to run on typical development hardware with reasonable resource consumption. Strict performance benchmarks are not required initially.
    *   **Output:** In `implementation_notes.md`, suggest including logging for training time and peak memory usage in the test script.
*   **CI/CD Integration:**
    *   This is **not a requirement** for the initial implementation. This can be noted briefly in `implementation_notes.md`.

## 5. Expected Outputs & Deliverables
As per `STAGE_GATE_PROMPT_PROG.md`, you are expected to:
1.  Create a new feature-specific subfolder within `/procedure/pnet_prostate_paper/roadmap/2_inprogress/`. This will likely be `09_pnet_training_integration_testing`.
2.  Move/copy the planning documents (`README.md`, `spec.md`, `design.md`) from the `1_planning` stage to this new `2_inprogress` folder.
3.  Inside this new `2_inprogress` subfolder, generate:
    *   `task_list.md`: A detailed breakdown of tasks required for implementation.
    *   `implementation_notes.md`: Notes covering the points outlined in Section 4 above, and any other relevant technical considerations for implementation.
4.  Ensure all paths used and generated are absolute paths.

## 6. Feedback Requirements
Upon completion of this task (successfully or with errors), you MUST create a detailed Markdown feedback file.
*   **Feedback File Location:** `/procedure/pnet_prostate_paper/roadmap/_active_prompts/feedback/`
*   **Feedback File Naming Convention:** `YYYY-MM-DD-HHMMSS-feedback-progress-pnet-integration-testing.md` (use the UTC timestamp of when you complete this task).
*   **Feedback File Content:**
    *   Reference this source prompt: `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-03-001825-progress-pnet-integration-testing.md`.
    *   Detail all actions taken (including folder creation/moving files).
    *   List all files created or modified, with their absolute paths.
    *   Describe any issues encountered and how they were handled.
    *   If you have any questions or need clarifications for the implementation phase, include them.

Please proceed with processing the feature into the in-progress stage.
