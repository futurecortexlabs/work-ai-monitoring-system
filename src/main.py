import yaml
from pathlib import Path

from camera import Camera


CONFIG_PATH = Path("config/sample.yaml")


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    print("===================================")
    print("Work AI Monitoring System Started")
    print("===================================")
    print(f"System  : {config['system']['name']}")
    print(f"Mode    : {config['system']['mode']}")
    print(f"Factory : {config['system']['factory_id']}")
    print(f"Line    : {config['system']['line_id']}")
    print("===================================")

    camera_config = config["camera"]

    camera = Camera(
        source=camera_config["source"],
        width=camera_config["width"],
        height=camera_config["height"],
        fps=camera_config["fps"],
    )

    try:
        camera.open()
        frame = camera.read()

        if frame is not None:
            print("Camera test: OK")
            print(f"Frame shape: {frame.shape}")
        else:
            print("Camera test: NG - No frame captured")

    except RuntimeError as e:
        print(f"Camera test: SKIPPED - {e}")

    finally:
        camera.release()


if __name__ == "__main__":
    main()