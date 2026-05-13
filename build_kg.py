import os
import re
from pathlib import Path


class Config:
    INPUT_DIR = Path("./data/西游记_章节分块")
    OUTPUT_DIR = Path("./data/KG")
    LOG_DIR = Path("./logs/KG_logs")
    ZHI_PU_API_KEY = os.environ.get("ZHI_PU_API_KEY")
    MODEL_NAME = "glm-4.5-air"
    MAX_CONCURRENT = 3

    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


def natural_keys(path):
    return [int(c) if c.isdigit() else c for c in re.split(r"([0-9]+)", path.name)]


class KGBuilder:
    def __init__(self):
        pass

    def process_file(self):
        files = sorted(Config.INPUT_DIR.glob("*.txt"), key=natural_keys)

        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            print(content)


if __name__ == "__main__":
    kg_builder = KGBuilder()
    kg_builder.process_file()
