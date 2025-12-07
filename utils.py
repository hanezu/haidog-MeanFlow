import yaml
from typing import Any

def load_yaml(path: str) -> Any:
    try:
        with open(path, 'r', encoding="utf-8") as file:
            return yaml.load(file, Loader=yaml.Loader)
    except yaml.YAMLError: 
        raise ValueError(rf"Invalid YAML format of file: '{path}'")

def dump_yaml(data: Any, path: str) -> None:
    with open(path, 'w', encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)