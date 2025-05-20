# AI Prompt: Stage Gate - From Idea to Plan

**Objective:** Transform a feature idea (from a `.md` file in `0_backlog/`) into a structured set of planning documents within a new dedicated subdirectory in `1_planning/`.

**Input:** The path to a `.md` file in the `0_backlog/` directory containing the feature idea.

**Process for AI Assistant:**

1.  **Identify Feature Name:**
    *   From the input backlog file path (e.g., `/procedure/pnet_prostate_paper/roadmap/0_backlog/FP001_feature_name.md`), extract a concise, directory-friendly name for the feature (e.g., `FP001_feature_name`). This will be the name of the new subdirectory.

2.  **Create Feature Directory:**
    *   Create a new subdirectory within `/procedure/pnet_prostate_paper/roadmap/1_planning/` using the extracted feature name (e.g., `/procedure/pnet_prostate_paper/roadmap/1_planning/FP001_feature_name/`).

3.  **Read and Understand the Idea File:**
    *   Thoroughly read the content of the input backlog `.md` file to understand the problem, goals, and any initial thoughts or requirements.

4.  **Generate Standard Planning Documents:**
    *   Inside the newly created feature directory, create the following standard planning documents. If templates are available in `/procedure/pnet_prostate_paper/roadmap/_templates/`, use them as a base. Otherwise, generate content based on the idea file and best practices.
        *   `README.md`: A high-level overview of the feature, its purpose, and status of planning.
        *   `SPEC.md` (Specification Document): Detailed functional and non-functional requirements. What should the feature do? What are the success criteria? Include user stories if applicable.
        *   `DESIGN.md` (Design Document): High-level technical design. How will the feature be implemented? What components are involved? Data models, API designs (if any), key algorithms, potential challenges, and considered alternatives.
        *   `PLAN.md` (Implementation Plan): Break down the feature into smaller, actionable tasks or development milestones. Estimate effort if possible. Identify dependencies.

5.  **Populate Documents:**
    *   Transfer relevant information from the input idea file into the appropriate sections of the newly created planning documents.
    *   Expand on the initial ideas, adding detail and structure as needed for each document type.
    *   Ensure traceability back to the original problem statement.

6.  **Update Original Backlog Item (Manual Step for USER):**
    *   After the AI has created the planning structure, the USER should manually move the original `.md` file from `/procedure/pnet_prostate_paper/roadmap/0_backlog/` to the newly created feature directory in `/procedure/pnet_prostate_paper/roadmap/1_planning/` to serve as a record of the initial idea.

**Output:**
*   A new subdirectory in `/procedure/pnet_prostate_paper/roadmap/1_planning/` containing `README.md`, `SPEC.md`, `DESIGN.md`, and `PLAN.md` populated with initial planning details for the feature.

**Example Invocation by USER to AI:**
"Execute `/procedure/pnet_prostate_paper/roadmap/1_planning/STAGE_GATE_PROMPT_PLAN.md` for the feature described in `/procedure/pnet_prostate_paper/roadmap/0_backlog/FP001_some_feature.md`."
