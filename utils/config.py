import os
from pathlib import Path


class Config:
    PROMPT_PATH = Path(__file__).parent / "prompts"
    TEXT_DIR = Path(__file__).parent.parent / "data"
    EXTRACT_DIR = Path(__file__).parent.parent / "extract"
    FILTER_DIR = Path(__file__).parent.parent / "filter"
    REFACTOR_DIR = Path(__file__).parent.parent / "refactor"
    Semaphore_Num = 3

class ModelConfig:
    API_KEY = os.environ.get("DEEPSEEK_KEY")
    BASE_URL = os.environ.get("DEEPSEEK_URL")
    MODEL_NAME = "deepseek-v4-flash"
