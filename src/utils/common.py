from box.exceptions import BoxValueError
from box import ConfigBox
import yaml
from pathlib import Path
from src import logger


def read_yaml(file_path: Path) -> ConfigBox:
    try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            return ConfigBox(content)
    except FileNotFoundError as e:
        logger.error(f"Exception: {e}")
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except yaml.YAMLError as e:
        logger.error(f"Exception: {e}")
        raise ValueError(f"Error reading YAML file: {file_path}") from e
    except BoxValueError as e:
        logger.error(f"Exception: {e}")
        raise ValueError(f"Error converting YAML to ConfigBox: {file_path}") from e
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise Exception(f"An unexpected error occurred: {file_path}") from e