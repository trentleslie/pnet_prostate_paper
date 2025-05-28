Hi Claude, we're very close to resolving all test failures for the P-NET model builders. One specific error remains, isolated to the attention mechanism in the `SparseTFSimple` layer.

**Goal:** Resolve the `tensorflow.python.framework.errors_impl.InvalidArgumentError` occurring in the `test_build_pnet2_with_attention` test within `model/testing/test_model_builders.py`.

**Background:**
Previous fixes successfully addressed general dimension mismatches in the `SparseTFSimple` layer and its interaction with mock data. All tests except `test_build_pnet2_with_attention` are now passing. The current error points to an issue with how indices are used within the attention-specific part of the `SparseTFSimple` layer.

**Error Details:**
-   **Error Message:** `tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error: indices[99] = [99, 49] does not index into shape [51,51]`
-   **Origin:** The error occurs in the `call` method of the `SparseTFSimple` layer (located in `model/layers_custom_tf2.py`, around line 297), specifically within the `ScatterNd` operation when `self.attention` is true (node name in traceback often includes `attention1_1/ScatterNd`).
-   **Failing Test:** `test_build_pnet2_with_attention`
-   **Context:**
    *   The `ScatterNd` operation is attempting to update a tensor of shape `[51,51]`. This is likely `self.kernel_attention` within `SparseTFSimple`, which is initialized with shape `(self.map.shape[1], self.map.shape[1])`. If `self.map.shape[1]` (number of output pathways from the current layer) is 51, then `self.kernel_attention` is indeed `(51,51)`.
    *   The operation fails because it's provided an index `[99, 49]`. The row index `99` is out of bounds for a tensor with 51 rows (valid indices 0-50).
    *   The `indices` argument for this `ScatterNd` is likely `self.map_sparse` (which is `tf.where(self.map > 0)` or similar, where `self.map` is the input-feature-to-pathway map, e.g., shape `(100, 51)`). If `self.map[99, 49]` is 1 (input feature 99 connects to output pathway 49), `tf.where` would produce `[99, 49]`.

**Hypothesis:**
The core issue seems to be a conceptual mismatch: indices derived from the input-feature-to-pathway map (`self.map_sparse`, e.g., `[input_idx, pathway_idx]`) are being incorrectly used to index into the pathway-to-pathway attention kernel (`self.kernel_attention`, shape `[num_pathways, num_pathways]`).

**Task Breakdown:**

1.  **Inspect `SparseTFSimple` Attention Logic:**
    *   Locate and carefully review the implementation of the `SparseTFSimple` class in `model/layers_custom_tf2.py`.
    *   Focus intensely on the `call` method, specifically the block executed `if self.attention:`. 
    *   Analyze how `self.kernel_attention` is constructed and updated using `tf.scatter_nd`.
    *   Confirm that `self.map_sparse` (or a similar variable holding indices from `self.map`) is being used as the `indices` argument for `tf.scatter_nd` when updating `self.kernel_attention`.
    *   Provide the relevant snippet of the `SparseTFSimple.call` method (the attention block) in your response.

2.  **Identify Root Cause of Indexing Mismatch:**
    *   Explain why using `self.map_sparse` (indices like `[input_feature_idx, pathway_idx]`) to update `self.kernel_attention` (a `[pathway_idx, pathway_idx]` matrix) is incorrect.

3.  **Propose and Implement a Fix:**
    *   Redesign or correct the logic for calculating or applying attention within `SparseTFSimple`.
    *   The attention mechanism should correctly relate pathways to each other, or input features to pathways, without using incompatible index spaces.
    *   Consider what the `self.kernel_attention` is intended to represent. If it's pathway-to-pathway attention, its update mechanism needs to be revised. If it's feature-to-pathway attention, the current indexing scheme is fundamentally flawed for a `[pathway, pathway]` kernel.
    *   Modify `model/layers_custom_tf2.py` with the corrected logic.

4.  **Verify:**
    *   Run the test suite `poetry run python -m unittest model/testing/test_model_builders.py`.
    *   Ensure that `test_build_pnet2_with_attention` passes and that no other tests regress. All 13 tests should pass.

**Key Files:**
*   `model/layers_custom_tf2.py` (Primary focus for the fix)
*   `model/testing/test_model_builders.py` (For running tests)
*   `model/builders/builders_utils.py` (Context for how `SparseTFSimple` is used, but likely no changes needed here for *this specific* error)

**Environment Requirement:**
All Python code execution and testing **must** be performed within the project's Poetry environment (e.g., `poetry run python -m unittest ...`).

Please investigate this attention mechanism carefully and implement the necessary corrections.
