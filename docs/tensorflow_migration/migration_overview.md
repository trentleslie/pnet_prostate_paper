# TensorFlow 1.x to 2.x Migration Overview

## Project Context

This document provides an overview of the migration of the P-NET (Pathway Network) codebase from TensorFlow 1.x to TensorFlow 2.x. The migration is necessary to ensure compatibility with Python 3.11 and modern TensorFlow APIs.

## Migration Status

The migration is being executed in phases, focusing on different components of the codebase:

1. **Keras Imports**: Updating imports from `keras.*` to `tensorflow.keras.*`
2. **TensorFlow API Calls**: Converting TF1.x patterns to TF2.x patterns
3. **Keras Model Definitions**: Refactoring `build_fn` functions
4. **Custom Callbacks**: Updating callback mechanisms for TF2.x
5. **Training Loops**: Ensuring compatibility with TF2.x Keras API

### Completed Work

- **Updated imports** in model modules
- **Refactored backend functions** in `get_gradient_layer` and related utilities
- **Created migration plans** for model building functions

### Current Focus

- **Model definition functions**: Refactoring `build_pnet2` and similar functions
- **Custom layers**: Updating `Diagonal` and `SparseTF` implementations
- **Testing strategy**: Ensuring refactored code behaves equivalently

## Key Files

The migration impacts several key files in the codebase:

- `/model/coef_weights_utils.py` - Gradient calculation utilities
- `/model/nn.py` - Core model handling class
- `/model/builders/prostate_models.py` - Model definitions
- `/model/builders/builders_utils.py` - Helper functions for model building
- `/model/layers_custom.py` - Custom layer implementations

## Next Steps

1. Complete refactoring of model definition functions
2. Update custom layer implementations
3. Validate numerical equivalence
4. Update training and evaluation code

## Migration Challenges

- **Eager Execution**: TF2.x uses eager execution by default, requiring different approaches for gradient calculation
- **Custom Layers**: Need to update for TF2.x compatibility
- **Session Management**: Need to eliminate `session` and `graph` operations
- **Model Loading**: Ensuring backward compatibility with saved models