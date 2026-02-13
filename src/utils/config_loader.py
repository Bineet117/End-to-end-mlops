import os
import yaml

class ConfigLoader:
    def __init__(self):
        self.config_dir = os.getenv("CONFIG_DIR", "configs")

    def load(self, name: str) -> dict:
        path = os.path.join(self.config_dir, f"{name}.yaml")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
