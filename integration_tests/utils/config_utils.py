import yaml
import os

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

def load_config(config_name: str) -> dict:
    """
    Loads a YAML configuration file from the 'configs' directory.

    Args:
        config_name (str): The name of the configuration file (e.g., "sample_test_config.yaml").

    Returns:
        dict: A dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    config_file_path = os.path.join(CONFIGS_DIR, config_name)
    
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    with open(config_file_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            # Add more context to the error if parsing fails
            error_msg = f"Error parsing YAML file: {config_file_path}\n{e}"
            # Check for common issues like scanner errors for more specific advice
            if hasattr(e, 'problem_mark'):
                error_msg += f"\nProblem at line {e.problem_mark.line+1}, column {e.problem_mark.column+1}"
            if hasattr(e, 'problem'):
                error_msg += f"\nProblem: {e.problem}"
            if hasattr(e, 'context'):
                error_msg += f"\nContext: {e.context}"
            raise yaml.YAMLError(error_msg)
    return config

if __name__ == '__main__':
    # Example usage:
    try:
        # Ensure 'sample_test_config.yaml' exists in the 'configs' directory for this example to work
        if os.path.exists(os.path.join(CONFIGS_DIR, 'sample_test_config.yaml')):
            sample_config = load_config("sample_test_config.yaml")
            print("Successfully loaded sample_test_config.yaml:")
            import json
            print(json.dumps(sample_config, indent=2))

            # Example of accessing a config value
            print(f"\nTest description: {sample_config.get('description')}")
            print(f"Dataset path: {sample_config.get('data', {}).get('dataset_path')}")
            print(f"Number of epochs: {sample_config.get('training', {}).get('epochs')}")
        else:
            print("sample_test_config.yaml not found. Skipping example usage.")

    except FileNotFoundError as e:
        print(e)
    except yaml.YAMLError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
