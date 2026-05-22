import yaml
from pathlib import Path


CONFIG_PATH = Path("config/sample.yaml")


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    system_name = config["system"]["name"]
    mode = config["system"]["mode"]
    factory_id = config["system"]["factory_id"]
    line_id = config["system"]["line_id"]

    print("===================================")
    print("Work AI Monitoring System Started")
    print("===================================")
    print(f"System  : {system_name}")
    print(f"Mode    : {mode}")
    print(f"Factory : {factory_id}")
    print(f"Line    : {line_id}")
    print("===================================")


if __name__ == "__main__":
    main()