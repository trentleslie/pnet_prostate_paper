Hi Claude, I'm working on unit tests for a Python-based P-NET (Pathway Network) model and encountering several failures after refactoring the test setup. The tests are written using the `unittest` framework.

**Goal:** Resolve all failing tests in `model/testing/test_model_builders.py`.

**Current Test Setup Overview:**
-   `MockData` class simulates `data.data_access.Data`.
-   `@patch('model.builders.prostate_models.Data', new=MockData)` is applied to the `TestModelBuilders` class to mock `Data` usage within `build_pnet` and `build_pnet2`.
-   `@patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)` is applied to `TestModelBuilders` to mock pathway data loading.

**Test Failures and Context:**

**1. `TypeError: ProstateDataPaper.__init__() got an unexpected keyword argument 'n_samples'` (Affects 6 original tests)**

   -   **Error Trace Snippet:**
       ```
       File "/procedure/pnet_prostate_paper/model/builders/prostate_models.py", line 231, in build_pnet2
         data = Data(**genomic_data_params)
       File "/procedure/pnet_prostate_paper/data/data_access.py", line 38, in __init__
         self.data_reader = ProstateDataPaper(**params)
       TypeError: ProstateDataPaper.__init__() got an unexpected keyword argument 'n_samples'
       ```
   -   **Problem:** This suggests that the *actual* `data.data_access.Data` class is being called instead of `MockData` for these original tests, despite the class-level patch. The `genomic_data_params` (passed to `Data`) contains an inner `'params'` dictionary, which includes `'n_samples'`. The real `Data` class passes this inner `'params'` dict to `ProstateDataPaper`, which doesn't accept `'n_samples'`.
   -   **Relevant Code:**
        *   `model/builders/prostate_models.py` imports `Data` as: `from data.data_access import Data` (Line 8). The patch target `@patch('model.builders.prostate_models.Data', new=MockData)` seems correct.
        *   `MockData.__init__`:
           ```python
           class MockData:
               last_instance = None
               def __init__(self, id="ALL", type="prostate_paper", params=None, ..., include_histology_features=False):
                   MockData.last_instance = self
                   self.params_arg = params if params is not None else {} # Stores the 'params' dict
                   # ... simulates data loading using self.params_arg.get('n_samples'), etc.
                   self.include_histology_features_received = include_histology_features
           ```
        *   `data.data_access.Data.__init__`:
           ```python
           class Data(object):
               def __init__(self, id="ALL", type="prostate_paper", params=None, ..., include_histology_features=None):
                   # ...
                   if type == "prostate_paper":
                       params_for_reader = params if params is not None else {}
                       self.data_reader = ProstateDataPaper(**params_for_reader) # Error occurs here
                   self.include_histology_features = include_histology_features if include_histology_features is not None else False
           ```
        *   `data.data_access.ProstateDataPaper.__init__`:
           ```python
           class ProstateDataPaper(object):
               def __init__(self, data_type=None, version='1_0', include_rppa=True): # Does not accept 'n_samples'
                   # ...
           ```
        *   Example of `data_params` in a failing test (e.g., `test_build_pnet2_basic`):
           ```python
           self.data_params = {
               'id': 'P1000', 'type': 'prostate_paper',
               'params': {'n_samples': 15, 'n_features': 10, 'n_genes': 5}, # This inner 'params' is the issue
               'version': '1_0', 'include_rppa': False
           }
           # build_pnet2 then does:
           # genomic_data_params = data_params.copy()
           # genomic_data_params['params'] = data_params.get('params', {}).copy()
           # data = Data(**genomic_data_params)
           ```
   -   **Question for Claude:** Why might the `@patch('model.builders.prostate_models.Data', new=MockData)` applied at the class level not be effective for these original tests, leading to the real `Data` class being called? How can this be fixed so `MockData` is consistently used?

**2. `AssertionError: False is not true` (Affects 2 new tests: `test_build_pnet_ignore_histology_false`, `test_build_pnet2_ignore_histology_false`)**

   -   **Error Trace Snippet:**
       ```
       File "/procedure/pnet_prostate_paper/model/testing/test_model_builders.py", line 358, in test_build_pnet_ignore_histology_false
         self.assertTrue(MockData.last_instance.include_histology_features_received)
       AssertionError: False is not true
       ```
   -   **Problem:** When `ignore_missing_histology=False` is passed to `build_pnet` or `build_pnet2`, the `MockData` instance still receives `include_histology_features=False`. It should receive `True`.
   -   **Relevant Code (from `model/builders/prostate_models.py` - applies to both `build_pnet` and `build_pnet2`):**
       ```python
       def build_pnetX(..., ignore_missing_histology=True):
           # ...
           if data_params is None: # data_params is an argument to build_pnetX
               data_params = {}
           genomic_data_params = data_params.copy()
           # Ensure 'params' key exists if it's expected by Data's __init__ for ProstateDataPaper
           if 'params' not in genomic_data_params:
                genomic_data_params['params'] = {}

           if ignore_missing_histology:
               genomic_data_params['include_histology_features'] = False
               logging.info('Building P-NET... model with genomic data only (histology ignored)')
           else: # This is when ignore_missing_histology is False
               logging.warning('ignore_missing_histology=False specified, but histology pathway not yet implemented. Using genomic data only.')
               genomic_data_params['include_histology_features'] = False # <<< THIS IS THE BUG
               logging.info('Building P-NET... model with genomic data only (histology pathway not implemented)')
           
           data = Data(**genomic_data_params) # MockData should be called here
           # ...
       ```
   -   **Question for Claude:** Please modify the logic in `build_pnet` and `build_pnet2` so that when `ignore_missing_histology=False`, `genomic_data_params['include_histology_features']` is set to `True`.

**3. `AssertionError: 1 != 2` in `test_get_pnet_basic` (Affects 1 original test)**

   -   **Error Trace Snippet:**
       ```
       File "/procedure/pnet_prostate_paper/model/testing/test_model_builders.py", line 407, in test_get_pnet_basic
         self.assertEqual(len(decision_outcomes), 2)
       AssertionError: 1 != 2
       ```
   -   **Problem:** The `test_get_pnet_basic` test expects two `decision_outcomes` when `n_hidden_layers=1`, but it's getting only one.
   -   **Relevant Code:**
        *   `test_get_pnet_basic` (from `model/testing/test_model_builders.py`):
           ```python
           def test_get_pnet_basic(self):
               # ... (setup of mock_data_instance, features, genes, ins) ...
               outcome, decision_outcomes, feature_names = get_pnet(
                   inputs=ins, features=features, genes=genes,
                   n_hidden_layers=1, # Critical for the assertion
                   direction='root_to_leaf', activation='tanh', activation_decision='sigmoid',
                   w_reg=0.01, w_reg_outcomes=0.01, # build_pnet passes w_reg as w_reg_outcomes
                   dropout=0.5, sparse=True, add_unk_genes=True, batch_normal=False,
                   kernel_initializer='glorot_uniform', use_bias=False, sparse_first_layer=True
               )
               self.assertEqual(len(decision_outcomes), 2)
           ```
        *   `_mock_get_layer_maps_for_test` (from `model/testing/test_model_builders.py`, used by `get_pnet` via `@patch`):
           ```python
           def _mock_get_layer_maps_for_test(genes, n_hidden_layers, direction='root_to_leaf', add_unk_genes=True):
               pathway_maps = {}
               num_mock_pathways_per_layer = [len(genes) // (i+1) if len(genes) // (i+1) > 0 else 1 for i in range(n_hidden_layers)]
               current_genes = list(genes)
               if add_unk_genes: current_genes.append('unk')

               for i in range(n_hidden_layers):
                   layer_map = {}
                   num_pathways = max(1, num_mock_pathways_per_layer[i])
                   pathway_names = [f'pathway_L{i}_{j}' for j in range(num_pathways)]
                   genes_per_pathway = len(current_genes) // num_pathways if num_pathways > 0 else len(current_genes)
                   if genes_per_pathway == 0 and len(current_genes) > 0: genes_per_pathway = 1
                   for k, name in enumerate(pathway_names):
                       start_idx = k * genes_per_pathway
                       end_idx = min((k + 1) * genes_per_pathway, len(current_genes))
                       assigned_genes = current_genes[start_idx:end_idx] if start_idx < len(current_genes) else []
                       layer_map[name] = assigned_genes
                   pathway_maps[f'layer{i+1}'] = layer_map
                   current_genes = pathway_names
               return pathway_maps
           ```
        *   `get_pnet` (from `model/builders/builders_utils.py` - snippet showing how `decision_outcomes` are populated):
           ```python
           def get_pnet(inputs, ..., n_hidden_layers, ...):
               # ... (initial layer1 setup, 'outcome' is output of layer1) ...
               decision_outcomes = []
               feature_names['h0'] = genes # features for the first hidden layer (gene layer)

               # Decision layer for the first hidden layer (h0)
               decision_layer0 = Dense(1, activation=activation_decision, kernel_regularizer=reg_l(w_reg_outcome0), name='decision_h0')(outcome)
               decision_outcomes.append(decision_layer0)
               feature_names['decision_h0'] = ['outcome_h0']

               # Loop for subsequent hidden layers (pathway layers)
               # pathway_maps = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes) # This is mocked
               pathway_maps = _mock_get_layer_maps_for_test(genes, n_hidden_layers, direction, add_unk_genes) # Effective call

               for i in range(n_hidden_layers): # This loop runs if n_hidden_layers > 0
                   # ... (constructs pathway_layer_i based on pathway_maps[f'layer{i+1}']) ...
                   # outcome = pathway_layer_i(outcome) # Output of current hidden layer
                   # decision_layer_i = Dense(1, activation=activation_decision, ..., name=f'decision_h{i+1}')(outcome)
                   # decision_outcomes.append(decision_layer_i)
                   # ...
               return outcome, decision_outcomes, feature_names
           ```
           The key part of `get_pnet` for `decision_outcomes` is:
           1. A `decision_layer0` is always added, connected to the output of the first gene-level layer (`layer1`). `decision_outcomes.append(decision_layer0)`.
           2. Then, a loop `for i in range(n_hidden_layers):` creates further hidden "pathway" layers. *Inside this loop*, another decision layer is typically added for each of these pathway layers.
           If `n_hidden_layers = 1`, the loop runs once (for `i=0`). This should add a `decision_layer1`. So, `decision_layer0` + `decision_layer1` = 2 outcomes.

   -   **Question for Claude:** Given the structure of `get_pnet` and `_mock_get_layer_maps_for_test`, why might `len(decision_outcomes)` be 1 instead of 2 when `n_hidden_layers=1`? Is there an issue in the `get_pnet` logic for appending decision outcomes, or how `_mock_get_layer_maps_for_test` provides data that `get_pnet` uses to build these layers?

Please provide code changes or explanations to address these three categories of errors.
