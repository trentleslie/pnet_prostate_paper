import logging

# Try to import pytest, but don't fail if it's not there when run directly
try:
    import pytest
except ImportError:
    if __name__ != "__main__": # Re-raise if not run directly
        raise
import os
import sys

# Add the parent directory of 'integration_tests' to sys.path
# to allow imports from 'utils' and other project modules.
# This assumes 'scripts' is a subdirectory of 'integration_tests'.
current_dir = os.path.dirname(os.path.abspath(__file__))
integration_tests_dir = os.path.dirname(current_dir)
project_root_dir = os.path.dirname(integration_tests_dir) # This should be pnet_prostate_paper
sys.path.insert(0, project_root_dir)
sys.path.insert(0, integration_tests_dir) # To find 'utils' directly if scripts is a package

from utils.logging_utils import setup_logging
from utils.config_utils import load_config # Import the config loader

# Setup logger for this test module
logger = setup_logging(log_file_name_prefix="test_setup", level=logging.INFO)

# Load a sample configuration for the test module
# This makes it available to all test functions in this file if needed.
try:
    test_config = load_config("sample_test_config.yaml")
    logger.info("Successfully loaded sample_test_config.yaml for test_integration_setup.")
except Exception as e:
    logger.error(f"Failed to load sample_test_config.yaml: {e}")
    test_config = None # Ensure test_config exists even if loading fails

def test_logging_and_config_setup():
    """
    Tests that logging and configuration loading are working.
    """
    logger.info("test_logging_and_config_setup: Logging is active.")
    if test_config:
        logger.info(f"Loaded config description: {test_config.get('description')}")
        logger.info(f"Data path from config: {test_config.get('data', {}).get('dataset_path')}")
        assert test_config.get('description') is not None
    else:
        logger.warning("test_logging_and_config_setup: test_config was not loaded.")
        assert False, "Test configuration could not be loaded."

if __name__ == '__main__':
    # This allows running the test file directly for debugging.
    # For actual test runs, pytest should be used.
    logger.info("Running test_integration_setup.py directly.")
    test_logging_and_config_setup()
    logger.info("Finished running test_integration_setup.py directly.")
    print(f"Log file for direct run should be in: {os.path.join(integration_tests_dir, 'logs')}")
    if 'pytest' not in sys.modules:
        print("Note: If 'pytest' module was not found, this run proceeded without it for basic checking.")
