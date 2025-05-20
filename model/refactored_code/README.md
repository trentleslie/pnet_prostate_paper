# Refactored Code for TensorFlow 2.x

This directory contains refactored implementations of key components for TensorFlow 2.x compatibility. These implementations serve as reference for the ongoing migration effort.

## Files

- `build_pnet2_refactored.py`: Refactored model building function
- `get_pnet_refactored.py`: Refactored network construction function
- `get_gradient_layer_refactored.py`: Refactored gradient calculation function

## Usage

These files are reference implementations and should not be imported directly in production code. Instead, they serve as templates for updating the actual implementation files in their respective locations.

Once tested and validated, the code from these files should be integrated into:

- `/procedure/pnet_prostate_paper/model/builders/prostate_models.py`
- `/procedure/pnet_prostate_paper/model/builders/builders_utils.py`
- `/procedure/pnet_prostate_paper/model/coef_weights_utils.py`

## Testing

Tests for these refactored implementations can be found in:

- `/procedure/pnet_prostate_paper/model/testing/`