from pathlib import Path
from enum import Enum
import json
import logging
import asyncio
from utils.config import Config, ModelConfig
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class method(Enum):
    extract = 1
    filter = 2
    refactor = 3


def call_model(client: AsyncOpenAI, system_prompt: str, text: str):
    message = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": text
        },
    ]
    response = client.chat.completions.create(model=ModelConfig.MODEL_NAME,
                                              messages=message,
                                              timeout=60)
    return response.choices[0].message.content


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=2, max=10))
async def async_call_model(client: AsyncOpenAI, system_prompt: str, text: str):
    message = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": text
        },
    ]
    response = await client.chat.completions.create(
        model=ModelConfig.MODEL_NAME,
        messages=message,
        timeout=60,
        response_format={'type': 'json_object'})
    return response.choices[0].message.content


def load_text(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def process_file(client: AsyncOpenAI, prompt: str,
                       logger: logging.Logger, semaphore: asyncio.Semaphore,
                       method: method):
    method_to_data = {
        method.extract: Config.TEXT_DIR,
        method.filter: Config.EXTRACT_DIR,
        method.refactor: Config.FILTER_DIR,
    }
    method_to_save = {
        method.extract: Config.EXTRACT_DIR,
        method.filter: Config.FILTER_DIR,
        method.refactor: Config.REFACTOR_DIR,
    }
    data_path = method_to_data.get(method)
    save_path = method_to_save.get(method)
    if data_path is None or save_path is None:
        raise ValueError("无效的 method")
    files = sorted(data_path.glob('*'))
    if not files:
        logger.warning(f"{method.name} 未找到任何文件")
    else:
        logger.info(f"{method.name} 找到 {len(files)} 个文件")

    async def process(file):
        async with semaphore:
            try:
                logger.info(f"正在处理 {file.name}")
                text = load_text(file)
                result = await async_call_model(client, prompt, text)
                output_path = save_path / f"{file.stem}.json"
                save_json(output_path, result)
                logger.info(f"保存结果到 {output_path}")
            except Exception as e:
                logger.error(f"文件{file.name}处理失败: {str(e)}")

    tasks = [process(file) for file in files]
    return await asyncio.gather(*tasks)


def save_json(path: Path, data):
    if isinstance(data, str):
        data = json.loads(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
