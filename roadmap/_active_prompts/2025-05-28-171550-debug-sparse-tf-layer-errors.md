Hi Claude, we've made progress on the unit tests for the P-NET model, but a new set of errors has emerged related to a custom sparse layer.

**Goal:** Resolve the `tensorflow.python.framework.errors_impl.InvalidArgumentError` occurring in 6 tests within `model/testing/test_model_builders.py`.

**Background:**
Previously, we fixed issues related to `MockData` patching, `ignore_missing_histology` logic, and `decision_outcomes` count. Now, several original tests for `build_pnet2` are failing during the `model.predict(X)` step.

**Error Details:**
-   **Error Message:** `tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error: indices[100] = [100, 100] does not index into shape [100,101]`
-   **Origin:** The error occurs in the `call` method of the `SparseTFSimple` layer (likely located in `model/layers_custom_tf2.py`, around line 297, possibly renamed from `SparseTF`). It's triggered by a `ScatterNd` operation.
-   **Failing Tests:**
    *   `test_build_pnet2_basic`
    *   `test_build_pnet2_single_output`
    *   `test_build_pnet2_with_attention`
    *   `test_build_pnet2_with_batch_norm`
    *   `test_build_pnet2_with_multiple_hidden_layers`
    *   `test_build_pnet2_with_sparse_options`

**Task Breakdown:**

1.  **Inspect `SparseTFSimple` Layer:**
    *   Locate and carefully review the implementation of the `SparseTFSimple` class in `model/layers_custom_tf2.py`. Pay close attention to:
        *   The `__init__` method: how it receives and processes pathway maps or connectivity information.
        *   The `build` method: how weights or other state are defined.
        *   The `call` method: how inputs are processed, especially the `ScatterNd` operation and how indices for it are generated. Note the shapes involved.
    *   Provide the source code of the `SparseTFSimple` class in your response for context.

2.  **Analyze Data Flow and Dimensions:**
    *   The error `indices[100] = [100, 100] does not index into shape [100,101]` suggests an off-by-one or dimension mismatch. The tensor being updated by `ScatterNd` appears to have a shape like `(100, 101)`, but an attempt is made to write to index `[100, 100]`, where the first `100` is out of bounds for the rows (0-indexed).
    *   Review how pathway maps (`pathway_maps`) and gene lists (`genes`, `features`) are prepared in `get_pnet` (in `model/builders/builders_utils.py`) and passed to `SparseTFSimple`.
    *   Review how the mock pathway maps are generated in `_mock_get_layer_maps_for_test` (in `model/testing/test_model_builders.py`), particularly focusing on dimensions and the effect of `add_unk_genes=True`.
    *   **Hypothesis:** The issue might stem from how the `add_unk_genes` flag (which adds an 'unknown' gene, potentially increasing a dimension by 1) interacts with the map creation in the mock function and its subsequent use in `SparseTFSimple`. The layer might expect dimensions consistent with `len(genes)` or `len(genes) + 1` depending on its internal logic for handling the pathway map.

3.  **Identify Root Cause:**
    *   Based on the inspection of `SparseTFSimple` and the data flow, pinpoint why the `ScatterNd` operation is receiving out-of-bounds indices.
    *   Determine if the issue lies in:
        *   The `SparseTFSimple` layer's internal logic.
        *   The way `get_pnet` preprocesses or passes data to `SparseTFSimple`.
        *   The dimensions or structure of the data produced by `_mock_get_layer_maps_for_test`.

4.  **Implement Fixes:**
    *   Modify the relevant file(s) (`model/layers_custom_tf2.py`, `model/builders/builders_utils.py`, or `model/testing/test_model_builders.py`) to correct the dimension mismatch or indexing logic.
    *   Ensure the fix is robust and correctly handles cases with and without `add_unk_genes`.

5.  **Verify:**
    *   Run the test suite `poetry run python -m unittest model/testing/test_model_builders.py` to confirm that all 13 tests now pass.

**Relevant Code Snippets (from previous sessions):**

*   **`_mock_get_layer_maps_for_test` (from `model/testing/test_model_builders.py` - this was significantly rewritten in the last feedback, please verify its current state in the file):**
    ```python
    # This is a placeholder, the actual implementation in the file might be different after recent fixes.
    # Claude should refer to the version in /procedure/pnet_prostate_paper/model/testing/test_model_builders.py
    def _mock_get_layer_maps_for_test(genes, pathway_config, direction='root_to_leaf', add_unk_genes=True):
        # ... (logic to generate mock pathway maps as list of DataFrames) ...
        # Example structure of pathway_config:
        # pathway_config = {
        # 'layer1_n_pathways': 5, 'layer1_name_prefix': 'L1_Pathway',
        # 'layer2_n_pathways': 3, 'layer2_name_prefix': 'L2_Pathway',
        # # ... and so on for n_hidden_layers
        # }
        # The function should return a list of pandas DataFrames, one for each hidden layer.
        # Each DataFrame should have columns like ['pathway_id', 'gene_id']
        # and represent the mapping for that layer.
        # The 'gene_id' in layer_i+1 should correspond to 'pathway_id' from layer_i.
        # Ensure dimensions account for `add_unk_genes`.
        pass # Actual implementation to be reviewed by Claude from the file
    ```

*   **`get_pnet` (from `model/builders/builders_utils.py` - snippet showing layer creation):**
    ```python
    def get_pnet(inputs, features, genes, n_hidden_layers, ..., add_unk_genes, sparse_first_layer, ...):
        # ...
        # pathway_maps_list_of_dfs = get_layer_maps(genes, pathway_config_for_mock, direction, add_unk_genes)
        # This is now effectively:
        # pathway_maps_list_of_dfs = _mock_get_layer_maps_for_test(genes, pathway_config_for_mock, direction, add_unk_genes)

        current_features = features
        current_genes_list = list(genes) # Original genes
        if add_unk_genes:
            current_genes_list.append('unk') # Add 'unk' gene if specified

        # First layer (gene layer)
        if sparse_first_layer: # This is True for the failing tests
            # Assuming Diagonal or a similar sparse layer is used for the first layer if sparse_first_layer is True
            # The output of this layer becomes 'outcome'
            layer1_output_dim = len(current_genes_list)
            # Example: layer1 = Diagonal(layer1_output_dim, ... name='h0')
            # outcome = layer1(inputs)
        else:
            # Dense layer
            pass

        # Subsequent hidden layers (pathway layers) using SparseTFSimple
        # for i in range(n_hidden_layers):
        #     pathway_map_df_for_layer_i = pathway_maps_list_of_dfs[i]
        #     # The 'map' argument for SparseTFSimple is derived from pathway_map_df_for_layer_i
        #     # The output dimension of SparseTFSimple is based on the number of unique pathways in pathway_map_df_for_layer_i
        #     # Example:
        #     # sparse_layer = SparseTFSimple(output_dim_for_layer_i, pathway_map_for_layer_i, ...)
        #     # outcome = sparse_layer(outcome)
        # ...
    ```

**Environment Requirement:**
All Python code execution, script running, and tool usage (like `pytest` or `unittest`) **must** be performed within the project's Poetry environment. This can typically be achieved by prefixing commands with `poetry run` (e.g., `poetry run python your_script.py`, `poetry run python -m unittest model/testing/test_model_builders.py`).

Please investigate thoroughly and provide the necessary code modifications.
