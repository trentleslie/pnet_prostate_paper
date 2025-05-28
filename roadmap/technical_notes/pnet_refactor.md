# P-NET Codebase Modernization: Python 3.11 & TensorFlow 2.x Upgrade

**Date:** 2025-05-28

## Executive Summary
This document provides a high-level overview of the ongoing project to modernize the P-NET (Prostate Network) codebase. The primary goal is to upgrade the system from its original Python 2.7 and TensorFlow 1.x foundation to Python 3.11 and TensorFlow 2.x. This modernization is crucial for enhancing the codebase's maintainability, performance, security, and compatibility with contemporary machine learning tools and libraries, ensuring the long-term viability and extensibility of the P-NET research platform. Recent milestones include the successful refactoring and comprehensive unit testing of core model building components.

## 1. Project Context & Motivation

The P-NET codebase, a valuable asset for prostate cancer research, was originally developed using Python 2.7 and TensorFlow 1.x. While groundbreaking at its inception, this technological foundation now presents several limitations:

*   **End-of-Life Technologies:** Python 2.7 reached its official end-of-life in 2020, meaning no further security updates or community support. TensorFlow 1.x is also considered legacy, with TensorFlow 2.x offering substantial improvements.
*   **Maintainability & Development Velocity:** Modern Python (3.11) and TensorFlow 2.x provide more intuitive APIs, better debugging capabilities (e.g., eager execution in TF2), and a richer ecosystem of supporting libraries, which can significantly improve developer productivity and code maintainability.
*   **Performance & Features:** TensorFlow 2.x offers performance enhancements and a more streamlined way to build and deploy models.
*   **Attracting Talent & Collaboration:** Utilizing an up-to-date tech stack makes the project more accessible and attractive for new researchers and collaborators.

This upgrade is essential to ensure P-NET remains a robust and relevant platform for cutting-edge research.

## 2. Key Challenges in Modernization

Migrating a complex research codebase across major versions of both its programming language and its core machine learning framework presents several interconnected challenges:

*   **Dual Upgrade Complexity:** Addressing simultaneous changes from Python 2 to 3 and TensorFlow 1 to 2 increases the scope and intricacy of the refactoring effort.
*   **Python 2.x to Python 3.x Transition:** This involves more than just syntax updates (e.g., `print` statements). Key areas include changes in string/bytes handling, integer division, and updates to standard library modules. While tools like `2to3` provide a starting point, manual review and adjustments are often necessary.
*   **TensorFlow 1.x to TensorFlow 2.x Migration:** This is the most substantial part of the technical challenge due to fundamental paradigm shifts:
    *   **API and Execution Model:** Moving from TensorFlow 1.x's graph-based, session-run execution model (using `tf.Session`, `K.function`) to TensorFlow 2.x's eager execution by default and `tf.function` for graph-based optimizations.
    *   **Keras Integration:** Standardizing on the core `tensorflow.keras` API, replacing older standalone Keras or `tf.contrib` usages.
    *   **Gradient Calculation:** Adapting code from `tf.gradients` or `optimizer.get_gradients` to use `tf.GradientTape`.
    *   **Custom Components:** Refactoring custom layers (like `SparseTFSimple` and its attention mechanism), callbacks (like the project's `GradientCheckpoint`), and utility functions to be compatible with the new TensorFlow APIs and execution model.
*   **Codebase Archeology:** Understanding and carefully refactoring legacy code, some of which may have limited original documentation or test coverage.
*   **Dependency Management:** Ensuring all third-party libraries are compatible with Python 3.11 and TensorFlow 2.x, and resolving any conflicts (e.g., the initial `kaleido` packaging issue).
*   **Testing and Validation:** The potential absence of comprehensive, automated test suites for the original codebase means that rigorous testing and validation are critical post-refactoring. This has involved developing new, detailed unit tests for core components.
*   **Missing Artifacts for Full End-to-End Testing:** While unit testing of model builders has advanced with robust mock data, the original `_logs/` directories and `*_params.yml` model configuration files are still needed for complete end-to-end testing of model loading, training, and evaluation pipelines with real data.

## 3. Strategic Approach & Brief Roadmap

Our approach to this modernization is phased to manage complexity and ensure steady progress:

*   **Phase 1: Foundational Python 3 Migration & Environment Setup (Completed)**
    *   Initial automated code conversion using `2to3`.
    *   Manual corrections and updates for Python 3 compatibility (e.g., print statements, standard library usage).
    *   Establishment of a modern Python development environment using Poetry for robust dependency management.
    *   Configuration of version control (Git) best practices.

*   **Phase 2: TensorFlow 2.x Core Component Refactoring (Completed)**
    *   **Core Utilities in `model/nn.py` (Completed):** Refactored methods like `get_layer_output` and `get_layer_outputs` to eliminate direct Keras backend calls (`K.function`, `K.learning_phase()`) and adopt TF2.x style (e.g., using temporary models for intermediate layer outputs). The `get_coef_importance` method is planned for a later stage.
    *   **Custom Keras Layers & Constraints (Completed):**
        *   Developed TF2.x compatible versions of `Diagonal` and `SparseTFSimple` (formerly `SparseTF`) layers within `layers_custom_tf2.py`.
        *   Implemented `SparseTFConstraint` for enforcing sparse connectivity, also in `layers_custom_tf2.py`.
        *   Updated model builder functions to import and use these TF2.x layers.
        *   Created and passed comprehensive unit tests for these layers.
    *   **Custom Keras Callbacks (`model/callbacks_custom.py`) (Partially Completed):** Updated `FixedEarlyStopping` and `GradientCheckpoint` for TF2.x Keras imports. The core logic of `GradientCheckpoint` appears compatible, but its reliance on an external `gradient_function` requires further verification for full TF2.x compatibility in an end-to-end training context.
    *   **Model Factory & Builder Function Adaptation (`model/model_factory.py`, `model/nn.py`) (Completed):** Significantly refactored the model factory to dynamically resolve `build_fn_name` strings (from YAML) to actual Python function objects. This includes creating Keras optimizer instances (e.g., `tf.keras.optimizers.Adam`) based on configuration and correctly structuring/passing parameters (including the new `ignore_missing_histology` flag) to `nn.Model` and subsequently to the specific builder functions.
    *   **Primary Model Building Functions & Test Suite (`model/builders/`, `model/testing/`) (Completed):**
        *   Successfully refactored primary model building functions (`build_pnet`, `build_pnet2` in `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`, and `get_pnet` in `/procedure/pnet_prostate_paper/model/builders_utils.py`) for TensorFlow 2.x compatibility.
        *   Resolved critical issues in the `SparseTFSimple` layer, particularly its attention mechanism, ensuring correct indexing and dimensional handling.
        *   Developed and/or significantly refactored mock data generation utilities (`_mock_get_layer_maps_for_test` in `/procedure/pnet_prostate_paper/model/testing/test_model_builders.py`) to provide robust inputs for unit tests.
        *   All unit tests (13/13) in `/procedure/pnet_prostate_paper/model/testing/test_model_builders.py` are now passing, validating the core model construction logic.

*   **Phase 3: Addressing Key Functionality & Critical Blockers (In Progress / Next Steps for specific items)**
    *   **`nn.Model.get_coef_importance` Method:** Investigating and refactoring this method, crucial for understanding model behavior. *Initial planning documents (`FP001`) have been created.*
    *   **Missing Parameter Files (`_params.yml`):** Strategy developed based on log analysis and an example `_params.yml` from historical logs. A `template_params.yml` and initial specific mock parameter files (`mock_basic_nn_params.yml`, `mock_nn_with_gradient_importance_params.yml`) have been created in `/procedure/pnet_prostate_paper/test_configs/mock_params/`. These currently point to full-sized data files. While robust mock data generation within unit tests has unblocked model builder validation, the creation of true minimal, self-contained test datasets and corresponding mock parameter files remains a task for enabling simplified end-to-end integration testing. *FP002 is in 2_inprogress.*

*   **Phase 4: Integration, Comprehensive Testing & Validation (Next Major Focus)**
    *   **Prerequisites:**
        *   The primary dataset (P1000 cohort) and external validation datasets referenced in Elmarakeby et al., Nature 2021 appear to be locally available within `/procedure/pnet_prostate_paper/data/_database/` (unzipped from `_database.zip`).
        *   Model construction components are now stable and unit-tested, providing a solid foundation for integration.
        *   Resolve the missing `_params.yml` files (as per roadmap item `FP002`) by creating robust mock parameter files that utilize the available local data or simplified mock data for initial integration tests.
    *   **Core Model Performance Replication:**
        *   Conduct end-to-end testing of the model's ability to predict disease state (primary vs. metastatic CRPC) using the main dataset files found in `/procedure/pnet_prostate_paper/data/_database/prostate/processed/`.
        *   Aim to replicate reported performance metrics (AUC, AUPRC, Accuracy, F1) utilizing the specific train/validation/test split files located in `/procedure/pnet_prostate_paper/data/_database/prostate/splits/` to align with the paper's methodology.
        *   Compare the refactored P-NET performance against baselines similar to those used in the paper (e.g., Logistic Regression, SVM, a simple dense neural network).
    *   **External Validation:**
        *   Utilize the external validation datasets found in `/procedure/pnet_prostate_paper/data/_database/prostate/external_validation/` to test the trained model and assess generalizability, comparing against reported classification accuracies.
    *   **Specific Analyses from the Paper to Consider for Replication:**
        *   **Biochemical Recurrence (BCR) Prediction:** If clinical outcome data linked to the patient samples is available, attempt to replicate the analysis correlating P-NET scores in primary tumors with BCR (referencing Fig. 2d in the paper).
        *   **Interpretability and Feature Importance:**
            *   Implement or integrate an attribution method compatible with TensorFlow 2.x (e.g., variants of DeepLIFT, SHAP, or Integrated Gradients) to analyze feature, gene, and pathway importance.
            *   Compare the highly-ranked features identified by the refactored model with those highlighted in the paper (e.g., AR, PTEN, TP53, MDM4, FGFR1).
    *   **Benchmarking and Regression Testing:**
        *   Develop new automated tests specific to the refactored codebase to ensure correctness and prevent regressions during ongoing development.
    *   **Note on Functional Validations:** The original paper includes extensive in vitro functional validations (e.g., CRISPR screens for MDM4). While these are typically beyond the scope of a codebase refactoring project, they provide crucial biological context for the model's findings and can inform potential downstream research directions leveraging the modernized P-NET.

*   **Phase 5: Documentation & Finalization (Future)**
    *   Updating all relevant technical documentation, including model descriptions and usage guides.
    *   Performing final codebase cleanup, review, and optimization.

## 4. Expected Outcomes & Benefits

Successfully completing this modernization project will yield significant benefits:

*   **Enhanced Maintainability:** A codebase that is easier to understand, modify, and extend.
*   **Improved Performance:** Potential for faster model training and inference with TensorFlow 2.x optimizations.
*   **Modern Tooling & Security:** Full compatibility with the latest Python and TensorFlow ecosystems, including security updates and access to new features.
*   **Increased Developer Productivity:** A more intuitive development experience with eager execution and simplified APIs.
*   **Long-Term Viability:** Ensuring the P-NET platform can continue to support cutting-edge research for years to come.
*   **Attraction for Collaboration:** A modern tech stack is more appealing for new researchers and collaborators.

## 5. Current Status (As of 2025-05-28)

*   The foundational Python 3 migration is **Completed**.
*   **Phase 2: TensorFlow 2.x Core Component Refactoring is Completed.** This includes:
    *   Core utilities in `model/nn.py` (e.g., `get_layer_output`) refactored for TF2.x.
    *   Custom Keras Layers (`Diagonal`, `SparseTFSimple`, `SparseTFConstraint`) are TF2.x compatible and unit-tested.
    *   Custom Keras Callbacks (`FixedEarlyStopping`, `GradientCheckpoint`) have updated imports; `GradientCheckpoint`'s `gradient_function` needs full TF2.x verification in an end-to-end context.
    *   The model factory (`model/model_factory.py`) and parameter handling (`model/nn.py`) have been substantially refactored for TF2.x.
    *   **Primary model building functions (`build_pnet`, `build_pnet2`, `get_pnet`) have been successfully refactored and their functionality validated through a comprehensive, passing unit test suite (`test_model_builders.py`). This includes fixes to the `SparseTFSimple` attention mechanism and mock data generation.**
*   **Phase 3: Addressing Key Functionality & Critical Blockers is In Progress** for specific items like `nn.Model.get_coef_importance` (`FP001`) and full `_params.yml` integration for end-to-end testing (`FP002`).
*   **The immediate next major focus is Phase 4: Integration, Comprehensive Testing & Validation.**

## Conclusion

The modernization of the P-NET codebase is a complex but essential undertaking. Significant progress has been made, with core components now refactored for Python 3.11 and TensorFlow 2.x, and validated through unit tests. While challenges remain, particularly concerning full end-to-end testing with original data configurations, the phased approach and detailed planning are designed to mitigate these risks. The successful completion of core component refactoring marks a major milestone, positioning the project for the critical next phase of integration testing and validation, which will ultimately result in a more robust, maintainable, and powerful research platform.