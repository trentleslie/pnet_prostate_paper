# Feedback: Debug SparseTF Layer Errors

**Date**: 2025-05-28  
**Time**: 17:22  
**Task**: Resolve `tensorflow.python.framework.errors_impl.InvalidArgumentError` in SparseTF layer affecting 6 tests

## Summary

Successfully resolved all SparseTF layer dimension mismatch errors that were causing `ScatterNd` index out-of-bounds failures during model prediction. All 13 tests in `model/testing/test_model_builders.py` now pass.

## Root Cause Analysis

### Primary Issue: Dimension Mismatch in SparseTF Layer

**Error**: `indices[100] = [100, 100] does not index into shape [100,101]`

**Root Causes Identified**:

1. **Incorrect Output Dimension Calculation**:
   - `get_pnet` was using `n_pathways = mapp.shape[0]` (rows) instead of `mapp.shape[1]` (columns)
   - This caused SparseTF layer to be created with wrong output dimension

2. **Mock Function Dimension Logic Error**:
   - Mock pathway maps had incorrect row count when `add_unk_genes=True`
   - Mock was adding UNK as an extra input feature instead of an output pathway
   - Resulted in 101×101 map when layer expected 100×N

### Data Flow Analysis

**Expected Flow**:
- h0 (Diagonal): Input(100) → Output(100)
- h1 (SparseTF): Input(100) → Output(N pathways)
- Map should be: 100 rows (input features) × N columns (pathways)

**Actual Flow Before Fix**:
- Mock created 101×101 map (21 genes + unk = 101 inputs)
- But h0 only outputs 100 features, not 101
- SparseTF kernel shape became (100, 101) but indices went up to [100, 100]
- Row index 100 was out of bounds for 100-row matrix

## Technical Fixes Applied

### 1. Fixed SparseTF Layer Creation

**File**: `model/builders/builders_utils.py`  
**Location**: Line 272

```python
# BEFORE (incorrect)
n_pathways = mapp.shape[0]  # Used row count as output dimension

# AFTER (correct)  
n_pathways = mapp.shape[1]  # Use column count as output dimension
```

**Impact**: Ensures SparseTF layer output dimension matches actual number of pathways.

### 2. Rewrote Mock Pathway Map Function

**File**: `model/testing/test_model_builders.py`  
**Function**: `_mock_get_layer_maps_for_test`

**Key Changes**:
- Fixed input dimension calculation to match actual layer inputs
- `add_unk_genes=True` now adds UNK pathway column, not input row
- Proper cascading: each layer's output becomes next layer's input
- Eliminated off-by-one dimension errors

**Before Fix Logic**:
```python
current_genes = list(genes)
if add_unk_genes:
    current_genes.append('unk')  # Wrong: adds input dimension
n_genes = len(current_genes)     # Results in 101 rows
```

**After Fix Logic**:
```python
current_input_features = list(genes)  # Use actual input dimensions
pathway_names = [f'pathway_L{i}_{j}' for j in range(num_pathways)]
if add_unk_genes:
    pathway_names.append('UNK')  # Correct: adds output pathway
```

## Verification Results

### Test Execution Results

**All 13 tests now pass**:
- ✅ `test_build_pnet2_basic`
- ✅ `test_build_pnet2_single_output`
- ✅ `test_build_pnet2_with_attention`
- ✅ `test_build_pnet2_with_batch_norm`
- ✅ `test_build_pnet2_with_multiple_hidden_layers`
- ✅ `test_build_pnet2_with_sparse_options`
- ✅ All histology and other tests from previous fixes

### Dimension Verification

**Single Layer (n_hidden_layers=1)**:
- Input: (None, 100)
- h0: (None, 100) 
- h1: (None, 51) - 100 inputs → 51 pathways (50 regular + 1 UNK)
- Map shape: (100, 51) ✅

**Multiple Layers (n_hidden_layers=2)**:
- h0: (None, 100)
- h1: (None, 51) - Map: (100, 51)
- h2: (None, 18) - Map: (51, 18)
- Proper cascading maintained ✅

## Technical Insights Gained

### 1. SparseTF Layer Architecture
- Input dimension = previous layer output dimension
- Output dimension = number of pathways (columns in connectivity map)
- `ScatterNd` requires indices within kernel shape bounds

### 2. Mock Function Design Principles
- Mock data structure must exactly match real function interface
- Dimension calculations must account for actual data flow, not theoretical gene counts
- `add_unk_genes` affects output pathways, not input features

### 3. TensorFlow Scatter Operations
- `tf.scatter_nd(indices, updates, shape)` strictly validates index bounds
- Indices must be within `[0, shape[i])` for each dimension
- Off-by-one errors cause hard failures, not warnings

## Quality Assurance

### Robustness Checks
- ✅ Single layer models work correctly
- ✅ Multiple layer models cascade properly
- ✅ Different pathway configurations (with/without UNK) work
- ✅ Various model options (attention, batch_norm, etc.) work
- ✅ Forward pass executes without errors

### Performance Impact
- No performance degradation observed
- Model compilation and prediction times remain normal
- Parameter counts are correct and reasonable

## Lessons Learned

### 1. Dimension Consistency is Critical
- Every layer transition must have matching input/output dimensions
- Mock functions must respect actual data flow, not abstract concepts

### 2. Index-Based Operations Require Careful Validation
- TensorFlow operations like `ScatterNd` have strict index validation
- Dimension mismatches manifest as runtime errors during prediction, not compilation

### 3. Test Infrastructure Design
- Mock functions should mirror real function behavior exactly
- Abstract simplifications can introduce subtle bugs
- Comprehensive testing across different configurations is essential

## Future Recommendations

1. **Enhanced Mock Validation**: Add dimension checking in mock functions to catch mismatches early
2. **Layer Dimension Logging**: Add debug logging for layer dimensions during model construction
3. **Automated Dimension Testing**: Create unit tests specifically for dimension consistency across layer transitions
4. **Documentation Updates**: Update layer documentation to clarify input/output dimension expectations

This debugging session successfully resolved all SparseTF layer errors and established a robust foundation for P-NET model testing with proper dimensional consistency.