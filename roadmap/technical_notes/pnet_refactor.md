# P-NET Codebase Modernization: Python 3.11 & TensorFlow 2.x Upgrade

**Date:** 2025-05-20

## Executive Summary
This document provides a high-level overview of the ongoing project to modernize the P-NET (Prostate Network) codebase. The primary goal is to upgrade the system from its original Python 2.7 and TensorFlow 1.x foundation to Python 3.11 and TensorFlow 2.x. This modernization is crucial for enhancing the codebase's maintainability, performance, security, and compatibility with contemporary machine learning tools and libraries, ensuring the long-term viability and extensibility of the P-NET research platform.

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
    *   **Custom Components:** Refactoring custom layers, callbacks (like the project's `GradientCheckpoint`), and utility functions to be compatible with the new TensorFlow APIs and execution model.
*   **Codebase Archeology:** Understanding and carefully refactoring legacy code, some of which may have limited original documentation or test coverage.
*   **Dependency Management:** Ensuring all third-party libraries are compatible with Python 3.11 and TensorFlow 2.x, and resolving any conflicts (e.g., the initial `kaleido` packaging issue).
*   **Testing and Validation:** The potential absence of comprehensive, automated test suites for the original codebase means that rigorous testing and validation are critical post-refactoring. This may involve developing new test cases.
*   **Missing Artifacts:** The current unavailability of crucial `_logs/` directories and `*_params.yml` model configuration files presents a significant blocker to end-to-end testing of model loading, training, and evaluation pipelines.

## 3. Strategic Approach & Brief Roadmap

Our approach to this modernization is phased to manage complexity and ensure steady progress:

*   **Phase 1: Foundational Python 3 Migration & Environment Setup (Largely Complete)**
    *   Initial automated code conversion using `2to3`.
    *   Manual corrections and updates for Python 3 compatibility (e.g., print statements, standard library usage).
    *   Establishment of a modern Python development environment using Poetry for robust dependency management.
    *   Configuration of version control (Git) best practices.

*   **Phase 2: TensorFlow 2.x Core Component Refactoring (In Progress)**
    *   **Keras Utilities & Callbacks:** Updating TensorFlow backend functions, custom callbacks (e.g., `GradientCheckpoint` refactoring to use `tf.GradientTape`). *Substantial progress has been made in this area.*
    *   **Model Building Functions:** Refactoring the core model definition scripts (e.g., `build_pnet2` in `prostate_models.py`, `get_pnet` in `model/builders_utils.py`) to utilize the `tensorflow.keras` API and TensorFlow 2.x patterns. *Detailed technical plans from AI assistant (Claude) are in place; implementation is the current primary focus.*
    *   **Custom Layers:** Ensuring any custom model layers (e.g., `Diagonal`, `SparseTF`) are updated for TensorFlow 2.x compatibility.

*   **Phase 3: Addressing Key Functionality & Critical Blockers (Planning / Next Steps)**
    *   **`nn.Model.get_coef_importance` Method:** Investigating and refactoring this method, crucial for understanding model behavior. *Initial planning documents (`FP001`) have been created.*
    *   **Missing Parameter Files (`_params.yml`):** Developing and implementing a strategy to handle the missing model configuration files. This may involve recovering them if possible, or creating mock/template files and a minimal test model setup to enable pipeline testing. *Initial planning documents (`FP002`) have been created.*

*   **Phase 4: Integration, Comprehensive Testing & Validation (Future)**
    *   **Prerequisites:**
        *   Secure [access to the datasets](https://zenodo.org/records/10775529) used in the original publication (e.g., Armenia et al. cohort, external validation cohorts referenced in Elmarakeby et al., Nature 2021). Refer to the paper's "Data availability" section and supplementary materials.
        *   Successfully resolve the missing `_params.yml` files (as per roadmap item `FP002`) to reconstruct original model configurations or establish well-justified new configurations based on paper descriptions.
    *   **Core Model Performance Replication:**
        *   Conduct end-to-end testing of the model's ability to predict disease state (primary vs. metastatic CRPC) using the main dataset.
        *   Aim to replicate reported performance metrics (AUC, AUPRC, Accuracy, F1) using the paper's 80%/10%/10% train/validation/test splits and/or 5-fold cross-validation methodology.
        *   Compare the refactored P-NET performance against baselines similar to those used in the paper (e.g., Logistic Regression, SVM, a simple dense neural network).
    *   **External Validation:**
        *   If external validation datasets (as mentioned in the paper) are accessible, test the trained model on these to assess generalizability, comparing against reported classification accuracies.
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

## 5. Current Status (As of 2025-05-20)

*   The foundational Python 3 migration is largely complete.
*   Significant progress has been made in refactoring core TensorFlow utilities and custom Keras callbacks, particularly the `GradientCheckpoint` mechanism.
*   Detailed technical plans, assisted by AI (Claude), are in place for the refactoring of primary model-building functions; this is the immediate next major coding task.
*   Planning has commenced for addressing the `nn.Model.get_coef_importance` functionality and the critical issue of missing `_params.yml` files, with dedicated roadmap items (`FP001`, `FP002`) established.

## Conclusion

The modernization of the P-NET codebase is a complex but essential undertaking. While challenges exist, particularly concerning the TensorFlow 1.x to 2.x migration and missing test artifacts, the phased approach and detailed planning are designed to mitigate these risks. The successful completion of this project will result in a more robust, maintainable, and powerful research platform, well-positioned for future advancements in prostate cancer research.