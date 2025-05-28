# Prompt: Critical Feedback on Proposed Python Unit Test Enhancements

**Date:** 2025-05-27
**Project:** P-NET TensorFlow 2.x Migration - Test Integration
**Source Prompt:** /procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-05-27-213100-feedback-on-test-plan.md
**Managed by:** Cascade (Project Manager AI)

## 1. Task Overview

We are integrating unit tests for a new `ignore_missing_histology` parameter added to two model building functions: `build_pnet` and `build_pnet2`. These functions are in `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`. The tests will be added to an existing test file: `/procedure/pnet_prostate_paper/model/testing/test_model_builders.py`.

The `ignore_missing_histology` parameter (defaults to `True`) controls whether histology data is considered.
- If `True` or default: only genomic data is used. `prostate_models.py` should log "Building P-NET... model with genomic data only (histology ignored)" and the `Data` class should be initialized with `include_histology_features=False`.
- If `False`: `prostate_models.py` should log a warning "ignore_missing_histology=False specified, but histology pathway not yet implemented..." and then "Building P-NET... model with genomic data only (histology pathway not implemented)". The `Data` class should be initialized with `include_histology_features=True` (which itself might trigger a warning in the `Data` class).

## 2. Existing Test Infrastructure (`test_model_builders.py`)

The target file `test_model_builders.py` already uses `unittest.TestCase` and has:
- A `MockData` class to simulate `data.data_access.Data`.
- Mocking for pathway data (`MockReactomeNetwork`, `mock_get_layer_maps`) to avoid external dependencies.
- Global patching: `pm.Data = MockData` (where `pm` is `model.builders.prostate_models`) and `bu.get_layer_maps = mock_get_layer_maps` (where `bu` is `model.builders.builders_utils`).
- A `TestModelBuilders(unittest.TestCase)` class with `setUp` and existing test methods.
- `setUp` defines `self.default_params` for `build_pnet2`.

Current `MockData.__init__` signature:
`def __init__(self, id="test", type="prostate_paper", params=None, test_size=0.3, stratify=True):`

Actual `Data.__init__` signature (from `data.data_access.py`):
`def __init__(self, id="ALL", type="prostate_paper", params=None, test_size=0.3, stratify=True, include_histology_features=False):`

The `build_pnet` and `build_pnet2` functions in `prostate_models.py` prepare a `data_params_dict` and call `Data(**data_params_dict)`. This `data_params_dict` will include `include_histology_features` as a key.

## 3. Proposed Changes for New Tests

### 3.1. Modification to `MockData.__init__`

**Goal:** `MockData` needs to accept `include_histology_features` as a keyword argument (like the real `Data` class) and record the value it received for assertion purposes.

**Proposed `MockData.__init__`:**
```python
# In MockData class
    def __init__(self, id="test", type="prostate_paper", params=None, 
                 test_size=0.3, stratify=True, include_histology_features=None): # Added include_histology_features
        """Initialize mock data, now also tracking include_histology_features."""
        self.id = id
        self.type = type
        # The 'params' arg to Data.__init__ is for feature selection, pathway versions etc.
        # It's distinct from other direct args like 'include_histology_features'.
        self.params_arg = params or {} # Renamed to avoid confusion with instance self.params used by old code
        self.test_size = test_size
        self.stratify = stratify
        self.include_histology_features_received = include_histology_features # Store it
        
        # Log how MockData was called regarding histology
        logging.info(f"MockData initialized. include_histology_features_received: {self.include_histology_features_received}")

        # Existing logic for dimensions, using self.params_arg if it contains them
        # This part needs to be careful: build_pnet passes data_params which contains an inner 'params' dict.
        # Data(**data_params) means MockData's 'params' arg will be data_params['params']
        # and 'include_histology_features' will be a direct kwarg.
        
        effective_params_for_dims = self.params_arg 
        
        if effective_params_for_dims and 'n_samples' in effective_params_for_dims:
            self.n_samples = effective_params_for_dims['n_samples']
        else:
            self.n_samples = 100 # Default
            
        if effective_params_for_dims and 'n_features' in effective_params_for_dims:
            self.n_features = effective_params_for_dims['n_features']
        else:
            self.n_features = 50 # Default
            
        if effective_params_for_dims and 'n_genes' in effective_params_for_dims:
            self.n_genes = effective_params_for_dims['n_genes']
        else:
            self.n_genes = 20 # Default
        
        self._generate_mock_data()
```

### 3.2. Example New Test Method (for `build_pnet2`, default case)

**Goal:** Verify logging and that `MockData` receives correct `include_histology_features` value.

**Proposed Test Method:**
```python
# In TestModelBuilders class (subclass of unittest.TestCase)
    def test_build_pnet2_ignore_histology_default(self):
        """Test build_pnet2 default behavior for ignore_missing_histology."""
        params = self.default_params.copy() # self.default_params is from setUp
        # No ignore_missing_histology specified in params, so build_pnet2 should default to True.
        # This means MockData should receive include_histology_features=False.
        
        with self.assertLogs(level='INFO') as cm: # Captures logs from root logger and children
            model, _ = build_pnet2(**params) # build_pnet2 is imported
        
        self.assertIsInstance(model, tf.keras.Model) # Basic model check
        
        # Check logs from prostate_models.py (model builder)
        self.assertTrue(any("Building P-NET2 model with genomic data only (histology ignored)" in log_msg for log_msg in cm.output))
        # Check logs from MockData.__init__
        self.assertTrue(any("MockData initialized. include_histology_features_received: False" in log_msg for log_msg in cm.output))

    # Other test methods would cover:
    # - build_pnet2 with ignore_missing_histology=True (MockData gets include_histology_features=False)
    # - build_pnet2 with ignore_missing_histology=False (MockData gets include_histology_features=True, specific warning logs)
    # - Similar set of tests for build_pnet function.
```

## 4. Request for Critical Feedback

Please provide critical feedback on the proposed approach, specifically:

1.  **`MockData` Modification:**
    *   Is the proposed change to `MockData.__init__` robust for capturing `include_histology_features` given how `build_pnet`/`build_pnet2` call `Data(**data_params_dict)`?
    *   Is the logging within `MockData` useful or just noise? Should it be `DEBUG` level?
    *   Any suggestions for improving how `MockData` handles or exposes the received `include_histology_features` for assertions (e.g., instead of checking logs)?

2.  **Test Method Structure:**
    *   Is the use of `self.assertLogs` appropriate and effective for checking the required log messages? 
    *   Are the assertions (`assertIsInstance`, log checks) sufficient for these tests? 
    *   The current approach relies on global patching of `pm.Data = MockData`. Is checking `MockData`'s log for `include_histology_features_received` a reliable way to assert what it received? Or is there a better way to inspect the `MockData` instance created *by the model builder within the test* (e.g., by modifying `MockData` to store its last instance, or using `unittest.mock.patch` with `side_effect` or `new_callable` to gain access to the instance)?

3.  **Parameter Handling for `build_pnet` vs `build_pnet2`:**
    *   `build_pnet` has a different signature than `build_pnet2` (e.g., `reg_outcomes` vs `w_reg_outcomes`, fewer overall parameters). Tests for `build_pnet` will require a separate "default_params_pnet1" dictionary in `setUp` or per test. Is this straightforward, or are there hidden complexities to consider for keeping tests DRY?

4.  **Potential Pitfalls & Improvements:**
    *   Are there any obvious pitfalls with this approach (e.g., test brittleness, over-reliance on log messages)?
    *   Are there alternative strategies for mocking or assertion that would be cleaner, more robust, or more idiomatic for Python's `unittest` framework?
    *   Any general improvements to make the tests more maintainable or clearer?

Please focus on identifying weaknesses, potential issues, and areas for improvement in the proposed test integration strategy.
