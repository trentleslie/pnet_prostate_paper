# Feedback: Strategies for `ignore_missing_histology` Implementation in `build_pnet`

**Date:** 2025-05-24 07:34:25 UTC  
**Source Prompt:** `/procedure/pnet_prostate_paper/roadmap/_active_prompts/2025-05-24-072800-strategize-ign-hist-buildpnet.md`  
**Task:** Strategic analysis of `ignore_missing_histology` parameter implementation

## Summary of Actions Taken

1. **Analyzed `build_pnet` function** in `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`
2. **Reviewed P-NET architecture** from the paper text to understand data input structure
3. **Examined parameter flow** through `model_factory.py` and `nn.py` to understand how `ignore_missing_histology` propagates
4. **Developed three distinct strategies** for handling missing histology data
5. **Analyzed implications** for each strategy in terms of architecture, performance, and implementation

## Analysis of Current `build_pnet` Implementation

### Current Architecture
The current `build_pnet` function accepts an `ignore_missing_histology` parameter but does not utilize it. The function:

- Uses `Data(**data_params)` to load molecular data (mutations, CNVs)
- Creates input tensor with shape `(n_features,)` where `n_features = x.shape[1]`
- Passes data through `get_pnet()` which builds a hierarchical pathway network
- According to the paper, P-NET processes "somatic mutation and copy number data" as primary inputs

### Data Input Structure (from Paper Analysis)
Based on the paper description:
- **Primary inputs:** Mutations and copy number alterations (CNVs) 
- **Optional inputs:** Fusions, gene expression (mentioned as feasible additions)
- **Histology data:** Not explicitly mentioned as a separate input type in the main architecture
- **Multi-modal approach:** The paper shows P-NET can integrate "multiple molecular features weighted differently"

### Parameter Flow Analysis
The `ignore_missing_histology` parameter flows as follows:
1. `model_factory.py` → `nn.Model()` constructor
2. `nn.py` → `Model.set_params()` method → `self.model_params['ignore_missing_histology']`
3. `nn.py` → `Model.fit()` → `self.build_fn(**self.model_params)` → `build_pnet()`

## Three Strategic Approaches for `ignore_missing_histology=True`

### Strategy 1: Input Data Filtering Approach
**Core Concept:** Modify the data loading/preprocessing step to exclude histology-related features from the input data when `ignore_missing_histology=True`.

**Implementation:**
```python
def build_pnet(optimizer, w_reg, add_unk_genes=True, sparse=True, dropout=0.5, 
               use_bias=False, activation='tanh', loss='binary_crossentropy', 
               data_params=None, n_hidden_layers=1, direction='root_to_leaf',
               batch_normal=False, kernel_initializer='glorot_uniform', 
               shuffle_genes=False, reg_outcomes=False, ignore_missing_histology=False):
    
    # Modify data_params to exclude histology features if needed
    if ignore_missing_histology:
        modified_data_params = data_params.copy()
        modified_data_params['exclude_histology'] = True
        data = Data(**modified_data_params)
    else:
        data = Data(**data_params)
    
    x, y, info, cols = data.get_data()
    # Continue with existing implementation...
```

### Strategy 2: Input Masking/Zeroing Approach
**Core Concept:** Allow histology features to remain in the input tensor but mask/zero them out before processing through the network.

**Implementation:**
```python
def build_pnet(..., ignore_missing_histology=False):
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    
    if ignore_missing_histology:
        # Identify histology feature indices (requires feature metadata)
        histology_indices = identify_histology_features(cols)
        # Create a mask to zero out histology features
        histology_mask = create_feature_mask(x.shape[1], histology_indices)
        
        # Apply masking layer early in the network
        ins = Input(shape=(n_features,), dtype='float32', name='inputs')
        masked_inputs = Lambda(lambda x: x * histology_mask, name='histology_mask')(ins)
        
        outcome, decision_outcomes, feature_n = get_pnet(masked_inputs, ...)
    else:
        ins = Input(shape=(n_features,), dtype='float32', name='inputs')
        outcome, decision_outcomes, feature_n = get_pnet(ins, ...)
```

### Strategy 3: Conditional Architecture Approach
**Core Concept:** Modify the P-NET architecture itself to conditionally exclude histology-related pathways and connections.

**Implementation:**
```python
def build_pnet(..., ignore_missing_histology=False):
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    
    # Filter genes and features based on histology exclusion
    if ignore_missing_histology:
        filtered_genes = filter_histology_genes(genes)
        filtered_features = filter_histology_features(features)
        n_features = len(filtered_features)
        n_genes = len(filtered_genes)
    else:
        filtered_genes = genes
        filtered_features = features
        n_features = x.shape[1]
    
    ins = Input(shape=(n_features,), dtype='float32', name='inputs')
    
    outcome, decision_outcomes, feature_n = get_pnet(ins,
                                                     filtered_features,
                                                     filtered_genes,
                                                     n_hidden_layers,
                                                     direction,
                                                     # ... other params
                                                     )
```

## Detailed Implications Analysis

### Strategy 1: Input Data Filtering Approach

**Impact on Layer Shapes:**
- Input layer dimensions reduced by the number of histology features
- All subsequent layer dimensions remain proportional
- Pathway mapping layers unaffected (unless histology-specific pathways exist)

**Data Flow Changes:**
- Clean separation at data loading stage
- No computational overhead during training/inference
- Maintains architectural integrity of P-NET

**Training & Performance Implications:**
- **Pros:** 
  - Fastest training (fewer input features)
  - Cleanest implementation
  - No architectural complexity
- **Cons:** 
  - Requires modification to `Data` class
  - Loss of potentially useful histology information permanently
  - May require data pipeline changes

**Interpretability:**
- High interpretability - clear which features are excluded
- Feature importance analysis unaffected for non-histology features
- Sankey diagrams and pathway analysis remain valid

**Implementation Risk:** Low - straightforward data preprocessing change

### Strategy 2: Input Masking/Zeroing Approach

**Impact on Layer Shapes:**
- Input layer dimensions unchanged
- All pathway layers maintain original structure
- Masking creates computational "dead zones" for histology features

**Data Flow Changes:**
- Histology features pass through as zeros
- May affect gradient flow through masked connections
- Potential for numerical instability if many features masked

**Training & Performance Implications:**
- **Pros:**
  - Maintains model architecture compatibility
  - Easy to toggle on/off during experimentation
  - No data pipeline changes required
- **Cons:**
  - Computational waste (processing zeros)
  - Potential gradient flow issues
  - May affect batch normalization statistics

**Interpretability:**
- Medium interpretability - masked features appear in feature lists but contribute nothing
- Feature importance analysis needs adjustment to handle masked features
- May confuse pathway analysis visualization

**Implementation Risk:** Medium - requires careful handling of masked gradients

### Strategy 3: Conditional Architecture Approach

**Impact on Layer Shapes:**
- Input layer dimensions reduced by histology features
- Potential reduction in pathway layer sizes if histology-specific pathways excluded
- Gene layer sizes may change based on histology gene filtering

**Data Flow Changes:**
- Fundamental change to network topology
- Pathway connections may be altered if histology pathways removed
- Feature mapping between layers changes

**Training & Performance Implications:**
- **Pros:**
  - Most biologically accurate approach
  - Optimal computational efficiency
  - Maintains pathway biological relevance
- **Cons:**
  - Complex implementation requiring pathway knowledge
  - May require separate pathway mapping logic
  - Potential for reduced model capacity

**Interpretability:**
- Highest interpretability - model truly reflects non-histology biology
- Feature importance and pathway analysis remain fully meaningful
- Sankey diagrams accurately represent active pathways

**Implementation Risk:** High - requires deep understanding of pathway mappings and gene classifications

## Recommendations

### Primary Recommendation: Strategy 1 (Input Data Filtering)
**Rationale:** 
- Cleanest implementation with lowest risk
- Maintains P-NET's architectural integrity
- Best performance characteristics
- Clear interpretability

**Implementation Priority:** High

### Secondary Recommendation: Strategy 3 (Conditional Architecture)
**Rationale:**
- Most biologically accurate
- Best long-term solution for comprehensive histology handling
- Highest scientific value

**Implementation Priority:** Medium (requires more development time)

### Not Recommended: Strategy 2 (Input Masking)
**Rationale:**
- Computational inefficiency
- Potential numerical issues
- Confusing interpretability
- No clear advantages over other approaches

## Questions for Project Manager (Cascade)

1. **Data Structure Clarification:** Is histology data currently included as separate features in the input data, or is this parameter anticipating future histology integration?

2. **Pathway Knowledge:** Do we have classification of which genes/pathways are histology-related vs. genomic/transcriptomic?

3. **Scope Definition:** Should `ignore_missing_histology` handle:
   - Missing histology data in existing features?
   - Complete exclusion of histology-related pathways?
   - Both scenarios?

4. **Implementation Timeline:** What is the priority for this implementation relative to the TensorFlow migration tasks?

## Next Steps Recommendation

1. **Clarify histology data structure** in current datasets
2. **Implement Strategy 1** as initial solution
3. **Plan Strategy 3** for comprehensive solution if histology pathway data is available
4. **Create unit tests** to validate ignore_missing_histology functionality across strategies