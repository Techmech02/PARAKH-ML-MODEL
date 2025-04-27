import yaml
import os

class Config:
    def __init__(self, config_path='config/config.yaml'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, section, key, default=None):
        """Access specific value from config"""
        try:
            return self.config[section][key]
        except KeyError:
            if default is not None:
                return default
            raise KeyError(f"Key '{key}' not found in section '{section}'")

    def get_section(self, section):
        """Access entire section"""
        try:
            return self.config[section]
        except KeyError:
            raise KeyError(f"Section '{section}' not found in config")

# Usage Example:
if __name__ == "__main__":
    cfg = Config()

    # Example usage
    print(cfg.get('paths', 'bloom_labeled_mcqs'))
    print(cfg.get('models', 'embedding_model'))
