# Implementation Plan: Strategy for Missing _params.yml Files / Test Data Generation

## 1. Phases & Tasks

**Phase 1: Investigation & Recovery (Est: 2-4 hours - Time-boxed)**

*   **Task 1.1:** Systematically check all potential sources for original `_logs/` data (Git history, other branches, team members, backups).
*   **Task 1.2:** Document findings. If files are found, validate them and integrate them into the workspace (if appropriate, respecting `.gitignore` if they are generated artifacts).

**Phase 2: Mock Parameter File System (If Task 1.1 Fails - Est: 3-5 hours)**

*   **Task 2.1:** Define the schema for `*_params.yml` by analyzing existing loader code (`DataModelLoader`, `model_factory`).
*   **Task 2.2:** Create the `/procedure/pnet_prostate_paper/test_configs/mock_params/` directory.
*   **Task 2.3:** Develop a `template_params.yml` with comments.
*   **Task 2.4:** Create 2-3 example mock `*_params.yml` files covering key test scenarios for `GradientCheckpoint`.
*   **Task 2.5:** Document how to use these mock files, including any necessary adjustments to scripts or environment to point to this directory.

**Phase 3: Minimal Test Model & Data (Optional, If Phase 2 is insufficient - Est: 2-3 hours)**

*   **Task 3.1:** Design a very simple Keras model for testing callbacks.
*   **Task 3.2:** Write a test script that defines this model, generates dummy data, and can run a minimal training loop to trigger callbacks.
*   **Task 3.3:** Integrate this into existing test suites or document its usage.

**Phase 4: Testing & Validation (Est: 2-3 hours)**

*   **Task 4.1:** Verify that models can be loaded using the mock parameter files.
*   **Task 4.2:** Confirm that `GradientCheckpoint` can be tested with different `feature_importance` settings using the mock files or the minimal test model.
*   **Task 4.3:** Ensure that relevant analysis scripts can be run (or adapted to run) with the new test configuration setup.

## 2. Dependencies

*   Outcome of the investigation (Phase 1).
*   Understanding of the `*_params.yml` structure from existing code.

## 3. Milestones

*   **M1:** Decision made on whether original `_logs/` data is recoverable.
*   **M2 (If mock files needed):** Mock parameter file system (`/procedure/pnet_prostate_paper/test_configs/mock_params/`) created with template and examples.
*   **M3 (If mock files needed):** Documentation for using mock files is complete.
*   **M4:** Key model functionalities (loading, `GradientCheckpoint`) can be tested using the established solution.
