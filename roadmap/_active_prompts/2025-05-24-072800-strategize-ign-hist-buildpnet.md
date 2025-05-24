# Prompt: Strategize `ignore_missing_histology` Implementation in `build_pnet`

**Date:** 2025-05-24
**Project:** P-NET TensorFlow 2.x Migration
**Source Prompt:** /procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-05-24-072800-strategize-ign-hist-buildpnet.md
**Managed by:** Cascade (Project Manager AI)

## 1. Task Overview

The goal of this task is to analyze the `build_pnet` model building function and propose robust strategies for incorporating the `ignore_missing_histology` parameter. This parameter, when true, should conditionally alter the P-NET model structure or input handling to account for missing histology data, while maintaining biological relevance as described in the original research.

## 2. Background and Context

The `ignore_missing_histology` boolean parameter is now passed down through the model factory (`model/model_factory.py`) and `model/nn.py` to the model builder functions. However, the primary builder, `build_pnet` (in `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`), does not yet utilize this parameter to modify its architecture.

Refer to the following for context:
*   **Original P-NET Paper:** `/procedure/pnet_prostate_paper/paper.txt` (especially sections describing the model architecture and data inputs).
*   **Model Builder Code:** `/procedure/pnet_prostate_paper/model/builders/prostate_models.py` (focus on `build_pnet`).
*   **Parameter Propagation:** `/procedure/pnet_prostate_paper/model/nn.py` (see `Model.set_params` and how `model_params` including `ignore_missing_histology` are passed to `build_fn`).
*   **Recent Status Update:** `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-05-22-tf2-core-refactoring-progress.md` (for an overview of recent changes and the importance of this task).

## 3. Specific Tasks & Deliverables

1.  **Analyze `build_pnet` and `paper.txt`:**
    *   Thoroughly review the existing `build_pnet` function in `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`.
    *   Correlate its structure with the P-NET architecture described in `/procedure/pnet_prostate_paper/paper.txt`, paying close attention to how histology data (and other data types like genomics, transcriptomics) are integrated.

2.  **Propose Strategies for `ignore_missing_histology=True`:**
    *   Develop 2-3 distinct, robust strategies for how `build_pnet` should modify the P-NET model architecture or data flow when `ignore_missing_histology` is `True`.
    *   Examples of modifications could include:
        *   Altering input layers (e.g., removing or providing zero/masked inputs for the histology pathway).
        *   Changing how different data pathways are concatenated or combined.
        *   Adjusting subsequent layer dimensions if an input pathway is effectively removed or altered.
    *   For each strategy, ensure it aligns with the biological rationale of P-NET where possible, or clearly state any deviations.

3.  **Detail Implications for Each Strategy:**
    *   For each proposed strategy, discuss:
        *   Impact on layer shapes and overall model complexity.
        *   Changes to data flow within the model.
        *   Potential implications for training, performance, and interpretability.
        *   Ease of implementation and potential risks.

## 4. Expected Output

The primary deliverable for this task is a detailed section within your feedback file outlining:
*   A brief summary of your analysis of `build_pnet` and relevant parts of the paper.
*   Each of the 2-3 proposed strategies for handling `ignore_missing_histology=True`.
*   For each strategy, the detailed implications as described in section 3.3.

This information will be used by the Project Manager (Cascade) and the USER to decide on the best approach before proceeding with implementation.

## 5. Environment and Execution

*   All Python code execution, script running, and tool usage (like `pytest`) **must** be performed within the project's Poetry environment. This can typically be achieved by prefixing commands with `poetry run` (e.g., `poetry run python your_script.py`) or by activating the shell environment using `poetry shell` before running commands.

## 6. Feedback Requirements

Upon completion of your analysis and strategy formulation, or if you encounter significant blockers or have clarifying questions:

1.  Create a feedback file named `YYYY-MM-DD-HHMMSS-feedback-strat-ign-hist-buildpnet.md` (e.g., `2025-05-24-153000-feedback-strat-ign-hist-buildpnet.md`, using UTC for timestamp when feedback is generated) in the `/procedure/pnet_prostate_paper/roadmap/_active_prompts/feedback/` directory.
2.  The feedback file should contain:
    *   A reference to this source prompt: `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-05-24-072800-strategize-ign-hist-buildpnet.md`.
    *   A summary of actions taken.
    *   The detailed strategies and implications as requested in Section 4 (Expected Output).
    *   Any issues encountered, assumptions made, or questions for the Project Manager (Cascade).

This prompt focuses solely on the strategic analysis. Implementation will be handled in a subsequent task.
