# Specification: Strategy for Missing _params.yml Files / Test Data Generation

## 1. Introduction
This document outlines the requirements for resolving the missing `*_params.yml` files issue, which currently blocks testing and model execution.

## 2. Functional Requirements

*   **FR1 (Investigation):** A thorough attempt MUST be made to locate and recover the original `_logs/` directory and its `*_params.yml` files.
*   **FR2 (If Unrecoverable - Mock Files):** If original files are unrecoverable, a system for using mock/template `*_params.yml` files MUST be established.
    *   **FR2.1:** Mock files MUST allow instantiation of `nn.Model` and other relevant model types.
    *   **FR2.2:** Mock files MUST contain necessary parameters for testing `GradientCheckpoint`, including various `feature_importance` settings (e.g., 'gradient', 'random', a callable function) and `feature_names`.
    *   **FR2.3:** A clear schema or example for creating these mock files MUST be documented.
*   **FR3 (If Unrecoverable - Test Data/Model):** If necessary, a minimal, self-contained test model and dummy dataset MAY be created to facilitate testing of callbacks and model mechanics without reliance on the full original data and complex configurations.
*   **FR4 (Documentation):** The chosen solution (recovered files, mock file strategy, or test model setup) MUST be documented clearly so team members can use it for testing.

## 3. Non-Functional Requirements

*   **NFR1:** The solution SHOULD allow for easy switching between different test configurations (e.g., different mock parameter files).
*   **NFR2:** Any new test data or mock files SHOULD be lightweight and not unnecessarily bloat the repository.

## 4. Success Criteria

*   Models can be instantiated using a defined set of parameter files (either original or mock).
*   The `GradientCheckpoint` callback can be tested with various `feature_importance` configurations.
*   The primary model loading and execution pathways are unblocked for testing.
*   Clear documentation exists on how to set up the testing environment regarding parameter files.
