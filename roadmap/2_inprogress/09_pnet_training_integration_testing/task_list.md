# Task List: P-NET Full Training Pipeline Integration Testing

## Phase 1: Data Preparation & Exploration
- [ ] Investigate original paper data sources in `/procedure/pnet_prostate_paper/paper.txt` for data availability details
- [ ] Examine existing data download scripts (`/data/prostate_paper/download_data.py`) to understand data format
- [ ] Analyze the minimal dataset creation script (`/scripts/create_minimal_dataset.py`) for reference
- [ ] Create or adapt a simplified dataset (100-500 samples) suitable for integration testing
- [ ] Implement data validation utilities to ensure correct format and shape

## Phase 2: Test Framework Setup
- [ ] Create new directory structure for integration tests (e.g., `/procedure/pnet_prostate_paper/integration_tests/`)
- [ ] Set up basic test infrastructure with proper imports and configuration
- [ ] Create configuration system for different test scenarios (YAML or Python config)
- [ ] Implement logging utilities for tracking test execution and results

## Phase 3: Model Building Integration
- [ ] Select initial P-NET model variant for testing (start with basic pnet with 1-2 sparse layers)
- [ ] Create wrapper functions for model instantiation using `get_pnet` from `/model/builders/builders_utils.py`
- [ ] Implement model configuration validation (verify architecture matches data)
- [ ] Add model summary/visualization utilities for debugging

## Phase 4: Training Pipeline Implementation
- [ ] Implement basic training loop with TensorFlow 2.x compatible code
- [ ] Add gradient monitoring utilities (check for NaN/Inf values)
- [ ] Implement metric collection (loss tracking, memory usage)
- [ ] Create checkpoint saving mechanism (optional for initial version)
- [ ] Add early stopping capabilities based on validation loss

## Phase 5: Validation & Health Checks
- [ ] Implement gradient flow validation (ensure all trainable layers receive gradients)
- [ ] Create loss convergence checks (verify loss decreases over epochs)
- [ ] Add memory usage monitoring
- [ ] Implement basic performance metrics (training time per epoch)
- [ ] Create comprehensive test report generation

## Phase 6: Test Case Development
- [ ] Create minimal smoke test (train for 2-3 epochs on tiny dataset)
- [ ] Develop standard integration test (full training on simplified real data)
- [ ] Add edge case tests (empty pathways, single sample, etc.)
- [ ] Implement parameterized tests for different model architectures

## Phase 7: Documentation & Integration
- [ ] Write comprehensive README for the integration test framework
- [ ] Document configuration options and test parameters
- [ ] Create usage examples and troubleshooting guide
- [ ] Integrate with existing test suite (if applicable)

## Phase 8: Refinement & Extension
- [ ] Address any issues discovered during initial testing
- [ ] Optimize test execution time for CI/CD compatibility
- [ ] Add support for additional P-NET variants (pnet2)
- [ ] Implement advanced validation metrics if needed

## Dependencies & Prerequisites
- TensorFlow 2.x environment
- Access to simplified prostate cancer dataset
- Functional P-NET model builders from recent debugging efforts
- Pathway map data (Reactome or similar)

## Success Criteria
- [ ] Integration tests run without crashes for at least 5 epochs
- [ ] Loss values show convergence trend (not NaN)
- [ ] All model layers receive valid gradients
- [ ] Test execution completes within reasonable time (<10 minutes)
- [ ] Clear test reports are generated
- [ ] Framework is easily extensible for future test cases