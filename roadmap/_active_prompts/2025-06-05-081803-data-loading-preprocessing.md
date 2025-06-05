# Task: Load and Preprocess P-NET Cancer Datasets

**Source Prompt Reference:** This task is defined by the prompt: `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-06-05-081803-data-loading-preprocessing.md`

## 1. Task Objective
To load and preprocess three key prostate cancer datasets (somatic mutation, copy number alteration, and clinical response) for the P-NET model. The outcome will be three cleaned, aligned, and validated pandas DataFrames: `mutation_df_processed`, `cna_df_processed`, and `response_df_processed`. This process assumes no histology data is being used, consistent with an `ignore_missing_histology=True` approach.

## 2. Prerequisites
- [X] Required files exist:
    - Somatic Mutation Data: `/procedure/pnet_prostate_paper/data/_database/prostate/processed/P1000_final_analysis_set_cross_important_only_plus_hotspots.csv`
    - Copy Number Alteration (CNA) Data: `/procedure/pnet_prostate_paper/data/_database/prostate/processed/P1000_data_CNA_paper.csv`
    - Response/Labels Data: `/procedure/pnet_prostate_paper/data/_database/prostate/processed/response_paper.csv`
- [X] Required permissions: Read access to the three data files listed above.
- [X] Required dependencies: Python 3.x, pandas library.
    - Installation (if needed): `pip install pandas` or `poetry add pandas`
- [X] Environment state: A Python environment where the pandas library can be imported and run.

## 3. Context from Previous Attempts (if applicable)
- **Previous attempt timestamp:** N/A (This is the first attempt for this specific structured prompt)
- **Issues encountered:** N/A
- **Partial successes:** N/A
- **Recommended modifications:** N/A

## 4. Task Decomposition
Break this task into the following verifiable subtasks:
1.  **Load Somatic Mutation Data (`mutation_df`):**
    *   Read `/procedure/pnet_prostate_paper/data/_database/prostate/processed/P1000_final_analysis_set_cross_important_only_plus_hotspots.csv` into a pandas DataFrame.
    *   Set the first column (sample identifiers) as the DataFrame index.
    *   Handle missing values (NaNs from empty CSV entries) by filling them with `0.0`.
    *   Ensure all data is numeric (float or int).
    *   Validation: Print shape and `.head()`; check for NaNs after fill; check dtypes.
2.  **Load CNA Data (`cna_df`):**
    *   Read `/procedure/pnet_prostate_paper/data/_database/prostate/processed/P1000_data_CNA_paper.csv` into a pandas DataFrame.
    *   Set the first column (sample identifiers) as the DataFrame index.
    *   Ensure all data is numeric.
    *   Validation: Print shape and `.head()`; check dtypes.
3.  **Load Response Data (`response_df`):**
    *   Read `/procedure/pnet_prostate_paper/data/_database/prostate/processed/response_paper.csv` into a pandas DataFrame.
    *   Set the 'Sample' column as the DataFrame index.
    *   Ensure the 'response' column is of integer type.
    *   Validation: Print shape and `.head()`; check 'response' column dtype.
4.  **Align Samples:**
    *   Identify common sample identifiers present across `mutation_df`, `cna_df`, and `response_df`.
    *   Filter all three DataFrames to retain only these common samples, ensuring consistent sample order.
    *   Validation: Print shapes of all three DataFrames after alignment; confirm they have the same number of rows.
5.  **Align Genes (Features):**
    *   Identify common gene names (columns) present in both `mutation_df` and `cna_df`.
    *   Filter `mutation_df` and `cna_df` to retain only these common genes, ensuring consistent gene order.
    *   Validation: Print shapes of `mutation_df` and `cna_df` after alignment; confirm they have the same number of columns.
6.  **Final Verification and Output:**
    *   Rename the processed DataFrames to `mutation_df_processed`, `cna_df_processed`, `response_df_processed`.
    *   Print the final shapes and display `.head()` for all three processed DataFrames.
    *   Validation: Confirm final DataFrames meet all criteria outlined in "Success Criteria and Validation".

## 5. Implementation Requirements
- **Input files/data:**
    - Mutation: `/procedure/pnet_prostate_paper/data/_database/prostate/processed/P1000_final_analysis_set_cross_important_only_plus_hotspots.csv`
    - CNA: `/procedure/pnet_prostate_paper/data/_database/prostate/processed/P1000_data_CNA_paper.csv`
    - Response: `/procedure/pnet_prostate_paper/data/_database/prostate/processed/response_paper.csv`
- **Expected outputs:** Three pandas DataFrames named `mutation_df_processed`, `cna_df_processed`, and `response_df_processed` available in the Python script's scope. The script should print their shapes and `.head()` to standard output.
- **Code standards:** Use pandas for data manipulation. Include clear comments for each step. Implement basic `try-except` blocks for file loading operations.
- **Validation requirements:** At each subtask, print relevant DataFrame shapes, `.info()`, or `.head()` to verify operations. Explicitly check for NaNs in `mutation_df_processed` after cleaning. Verify data types of key columns (e.g., 'response' column).

## 6. Error Recovery Instructions
If you encounter errors during execution:
- **Permission/Tool Errors (e.g., `FileNotFoundError`, `PermissionError`):**
    - Verify the full absolute paths to the input CSV files are correct.
    - Ensure the user/process running the script has read permissions for these files.
    - Classification: RETRY_WITH_MODIFICATIONS (after path/permission correction).
- **Dependency Errors (e.g., `ModuleNotFoundError: No module named 'pandas'`):
    - Instruct the user to install pandas: `pip install pandas` or `poetry add pandas`.
    - Classification: RETRY_WITH_MODIFICATIONS (after dependency installation).
- **Configuration Errors (e.g., CSV parsing issues, incorrect delimiter, unexpected headers):
    - Suggest inspecting the CSV files manually to confirm format (delimiter, header presence).
    - For pandas `read_csv`, suggest trying `sep=','` explicitly if default doesn't work, or `header=0`.
    - Classification: RETRY_WITH_MODIFICATIONS or ESCALATE_TO_USER if format is ambiguous.
- **Logic/Implementation Errors (e.g., KeyError during alignment, unexpected DataFrame shapes):
    - Suggest printing `.columns`, `.index`, and `.shape` of DataFrames before the failing operation.
    - Double-check column/index names used for merging or filtering.
    - Classification: RETRY_WITH_MODIFICATIONS or ESCALATE_TO_USER.

For each error type, indicate in your feedback:
- Error classification: [RETRY_WITH_MODIFICATIONS | ESCALATE_TO_USER | REQUIRE_DIFFERENT_APPROACH]
- Specific changes needed for retry (if applicable)
- Confidence level in proposed solution

## 7. Success Criteria and Validation
Task is complete when:
- [ ] `mutation_df_processed` is created: indexed by sample ID, columns are gene names, values are numeric (0.0 or 1.0), and contains no NaN values.
- [ ] `cna_df_processed` is created: indexed by sample ID, columns are gene names, values are numeric.
- [ ] `response_df_processed` is created: indexed by sample ID, contains a 'response' column of integer type (0 or 1).
- [ ] All three processed DataFrames (`mutation_df_processed`, `cna_df_processed`, `response_df_processed`) have the same number of rows (samples) and their indices (sample IDs) are aligned.
- [ ] `mutation_df_processed` and `cna_df_processed` have the same number of columns (genes) and their column names (gene names) are aligned.
- [ ] The shapes of the original DataFrames (after initial load) and the final processed DataFrames are printed to standard output.
- [ ] The first 5 rows (`.head()`) of each final processed DataFrame are printed to standard output.

## 8. Feedback Requirements
Create a detailed Markdown feedback file at:
`[PROJECT_ROOT]/roadmap/_active_prompts/feedback/YYYY-MM-DD-HHMMSS-feedback-data-loading-preprocessing.md`
(Replace `[PROJECT_ROOT]` with the actual project root, and `YYYY-MM-DD-HHMMSS` with the execution timestamp in UTC).

**Mandatory Feedback Sections:**
- **Execution Status:** [COMPLETE_SUCCESS | PARTIAL_SUCCESS | FAILED_WITH_RECOVERY_OPTIONS | FAILED_NEEDS_ESCALATION]
- **Completed Subtasks:** [List subtasks from section 4 that were successfully completed]
- **Issues Encountered:** [Detailed error messages, tracebacks, and context for any issues]
- **Next Action Recommendation:** [e.g., Proceed to model building, Retry with specific changes, Escalate to user with questions]
- **Confidence Assessment:** [Your confidence in the correctness and completeness of the processed data (High/Medium/Low) and why]
- **Environment Changes:** [Any files created/modified (other than feedback), dependencies installed, etc.]
- **Lessons Learned:** [Any insights gained, patterns that worked well, or potential improvements for similar tasks]
- **Output Snippets:** Include printed shapes and `.head()` outputs for final DataFrames as part of the feedback or confirm they were printed to stdout.
