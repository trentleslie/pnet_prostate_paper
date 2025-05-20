# Testing Directory for TensorFlow 2.x Migration

This directory contains test scripts for validating the TensorFlow 2.x refactoring of the P-NET codebase.

## Test Files

- `test_gradient_layer.py`: Tests for the refactored gradient calculation functions

## Running Tests

To run the tests, use:

```bash
python -m procedure.pnet_prostate_paper.model.testing.test_gradient_layer
```

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a clear, descriptive name for the test file
2. Include both simple and complex test cases
3. Compare outputs with original implementations
4. Document any numerical differences or edge cases
5. Add the test to this README

## Validation Metrics

For gradient computation comparisons, the tests use:

1. Correlation between TF1.x and TF2.x outputs
2. Mean absolute difference
3. Mean relative difference
4. Visual comparison of distributions