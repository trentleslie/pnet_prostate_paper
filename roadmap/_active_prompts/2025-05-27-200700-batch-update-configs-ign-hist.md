# Prompt: Batch Update Python Config Scripts for `ignore_missing_histology`

**Date:** 2025-05-27
**Project:** P-NET TensorFlow 2.x Migration
**Source Prompt:** {{PROJECT_ROOT}}/roadmap/_active_prompts/2025-05-27-200700-batch-update-configs-ign-hist.md
**Managed by:** Cascade (Project Manager AI)

## 1. Task Overview

The goal is to update multiple Python-based model configuration scripts to explicitly include the `'ignore_missing_histology': True` parameter. This parameter ensures that the P-NET models (`build_pnet` or `build_pnet2`) are configured to run using only genomic data, which is the current standard behavior.

## 2. Background and Context

The `build_pnet` and `build_pnet2` functions in `/procedure/pnet_prostate_paper/model/builders/prostate_models.py` now include an `ignore_missing_histology` parameter, which defaults to `True`. To make this explicit in existing configuration scripts, we need to add this key-value pair to the parameter dictionaries passed to these functions.

The common structure in these configuration scripts involves a main dictionary (often named `nn_pathway` or similar) containing a sub-dictionary under the key `'params'`, which in turn contains:
- A `'build_fn'` key (e.g., `'build_fn': build_pnet2`).
- A `'model_params'` key, whose value is a dictionary containing the parameters for the model building function. This is the target dictionary for our modification.

Example structure:
```python
# config_script.py
# ... imports and other variables ...

nn_pathway = {
    'type': 'nn',
    'id': 'P-net',
    'params': {
        'build_fn': build_pnet2, # Or build_pnet
        'model_params': {      # <-- THIS IS THE TARGET DICTIONARY
            'use_bias': True,
            'w_reg': wregs,
            # ... other existing parameters ...
            # We need to add 'ignore_missing_histology': True here
        },
        'fitting_params': {
            # ...
        }
    }
}
# ...
models = [nn_pathway]
# ...
```

## 3. Specific Task: Modify Python Configuration Scripts

You are to process the list of Python files provided below. For each file:

1.  **Parse the Python code** to locate the target `model_params` dictionary.
    *   The target `model_params` dictionary is the one associated with a `build_fn` that is either `build_pnet` or `build_pnet2`.
    *   Be robust to variations in the name of the parent dictionary (e.g., it might not always be `nn_pathway`). The key is the structure: a dictionary containing `'build_fn'` and `'model_params'`.

2.  **Check for Existing Key:**
    *   If the `'ignore_missing_histology'` key already exists within the identified `model_params` dictionary, do nothing to this dictionary and note that the file is already compliant or was previously modified.

3.  **Add Key if Missing:**
    *   If the key is not present, add `'ignore_missing_histology': True,` to the `model_params` dictionary.
    *   Preserve the original formatting (indentation, style) of the file as much as possible.
    *   A good placement for the new key would be after the last existing parameter in the dictionary, before the closing brace `}`. Ensure a trailing comma if it's not the absolute last item in a multi-line dictionary definition.

4.  **Handle Multiple Models:**
    *   Some configuration scripts might define multiple models (e.g., in a list assigned to `models`). Ensure you correctly identify and modify the `model_params` for each P-NET model definition (`build_pnet` or `build_pnet2`) within the file.

5.  **Error Handling & Reporting:**
    *   If a file cannot be parsed, or the expected structure is not found, skip the file and report the issue.
    *   Do not modify files that do not fit the expected pattern for P-NET model configuration.

## 4. List of Files to Process:

```
/procedure/pnet_prostate_paper/train/params/P1000/review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_cnv_burden2.py
/procedure/pnet_prostate_paper/train/params/P1000/review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_TMB_cnv.py
/procedure/pnet_prostate_paper/train/params/P1000/review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_TMB.py
/procedure/pnet_prostate_paper/train/params/P1000/review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_TMB2.py
/procedure/pnet_prostate_paper/train/params/P1000/review/crossvalidation_average_reg_10_tanh_cancer_genes.py
/procedure/pnet_prostate_paper/train/params/P1000/review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_account_zero.py
/procedure/pnet_prostate_paper/train/params/P1000/review/onsplit_average_reg_10_tanh_large_testing_ge.py
/procedure/pnet_prostate_paper/train/params/P1000/review/LOOCV_reg_10_tanh.py
/procedure/pnet_prostate_paper/train/params/P1000/review/10custom_arch/onsplit_kegg.py
/procedure/pnet_prostate_paper/train/params/P1000/dense/onsplit_dense.py
/procedure/pnet_prostate_paper/train/params/P1000/review/9hotspot/onsplit_average_reg_10_tanh_large_testing_count.py
/procedure/pnet_prostate_paper/train/params/P1000/review/fusion/onsplit_average_reg_10_tanh_large_testing_inner_fusion_genes.py
/procedure/pnet_prostate_paper/train/params/P1000/review/fusion/onsplit_average_reg_10_tanh_large_testing_TMB.py
/procedure/pnet_prostate_paper/train/params/P1000/review/fusion/onsplit_average_reg_10_tanh_large_testing_fusion_zero.py
/procedure/pnet_prostate_paper/train/params/P1000/review/fusion/onsplit_average_reg_10_tanh_large_testing_fusion.py
/procedure/pnet_prostate_paper/train/params/P1000/review/learning_rate/onsplit_average_reg_10_tanh_large_testing_inner_LR.py
/procedure/pnet_prostate_paper/train/params/P1000/review/9single_copy/onsplit_average_reg_10_tanh_large_testing_single_copy.py
/procedure/pnet_prostate_paper/train/params/P1000/review/9single_copy/crossvalidation_average_reg_10_tanh_single_copy.py
/procedure/pnet_prostate_paper/train/params/P1000/external_validation/pnet_validation.py
/procedure/pnet_prostate_paper/train/params/P1000/pnet/crossvalidation_average_reg_10_tanh.py
/procedure/pnet_prostate_paper/train/params/P1000/review/9hotspot/onsplit_average_reg_10_tanh_large_testing_hotspot.py
/procedure/pnet_prostate_paper/train/params/P1000/review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_cnv_burden.py
/procedure/pnet_prostate_paper/train/params/P1000/review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_account_zero2.py
/procedure/pnet_prostate_paper/train/params/P1000/number_samples/crossvalidation_average_reg_10.py
/procedure/pnet_prostate_paper/train/params/P1000/number_samples/crossvalidation_average_reg_10_tanh.py
/procedure/pnet_prostate_paper/train/params/P1000/pnet/onsplit_average_reg_10_tanh_large_testing.py
```
*(Note: The file `/procedure/pnet_prostate_paper/train/params/P1000/pnet/onsplit_average_reg_10_tanh_large_testing_inner.py` was already manually updated and is intentionally omitted from this list, but you can report if you encounter it and find it compliant).*

## 5. Expected Output

*   **Modified Files:** Apply changes directly to the files.
*   **Feedback File:** Create a feedback file named `YYYY-MM-DD-HHMMSS-feedback-batch-update-configs.md` in the `{{PROJECT_ROOT}}/roadmap/_active_prompts/feedback/` directory. This file should detail:
    *   A list of files successfully modified.
    *   A list of files that already contained the `'ignore_missing_histology': True` key and were not changed.
    *   A list of files that could not be processed (with reasons, e.g., structure not found, parsing error).
    *   Any assumptions made or difficulties encountered.
*   **IMPORTANT:** If you are unable to write the feedback file to the specified path for any reason, please print the COMPLETE intended content of the feedback file to standard output before terminating.

## 6. Environment and Execution

*   All Python code execution, script running, and tool usage **must** be performed within the project's Poetry environment: `poetry run ...` or `poetry shell`.
*   The `claude` command, if needed by you for auxiliary tasks, requires `nvm use --delete-prefix v22.15.1 && claude ...`.
