import os
import json
import logging
import asyncio
import aiofiles
import time
from datetime import datetime
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)


# --- 配置 ---
class Config:
    INPUT_DIR = Path("./data/西游记_章节分块")
    OUTPUT_DIR = Path("./data/Extracted")
    LOG_DIR = Path("./logs/Extract_logs")
    ZHI_PU_API_KEY = os.environ.get("ZHI_PU_API_KEY")
    MODEL_NAME = "glm-4.5-air"
    MAX_CONCURRENT = 3

    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


Config.setup()

# --- 日志配置 ---
log_filename = datetime.now().strftime("extract_%Y%m%d_%H%M%S.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_DIR / log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

system_prompt = """
# Role
你是一个精通中文文学的文本提取助手。

# Task
提取文档中[孙悟空]（及其别名：石猴、美猴王、猴王、大圣）的所有原话对话。

# Rules
1. 仅提取引号内的对话原文。
2. 排除旁白（例如：不要包含“那猴王道：”这些字）。
"""


# --- 数据模型 ---
class DialogueExtraction(BaseModel):
    quotes: list[str] = Field(description="该角色说的具体话语列表")


class Extract:
    def __init__(self):
        self.model = ChatZhipuAI(
            model=Config.MODEL_NAME,
            api_key=Config.ZHI_PU_API_KEY,
            temperature=0.1,
            timeout=600,
        ).with_structured_output(DialogueExtraction)

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{text}")]
        )

        self.chain = self.prompt | self.model
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def call_llm(self, text: str) -> DialogueExtraction:
        """LLM调用"""
        return await self.chain.ainvoke({"text": text})

    async def extract_dialogue(self, file_path: Path) -> dict:
        """处理单个文件的异步任务"""

        async with self.semaphore:
            start_t = time.perf_counter()
            report = {
                "filename": file_path.name,
                "status": "FAILED",
                "count": 0,
                "duration": 0,
            }
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                if not content.strip():
                    logger.warning(f"文件{file_path.name}为空,跳过此文件")
                    report["status"] = "SKIPPED"
                    return report

                logger.info(f"正在提取: {file_path.name} ({len(content)} 字)...")
                result: DialogueExtraction = await self.call_llm(content)
                if result and result.quotes:
                    await self.save_to_json(file_path, result)
                    report.update(
                        {
                            "status": "SUCCESS",
                            "count": len(result.quotes),
                            "duration": time.perf_counter() - start_t,
                        }
                    )
                    logger.info(f"文件：{file_path.name}(提取{report['count']})")
                else:
                    logger.warning(f"文件{file_path.name}未提取到对话")
                    report["status"] = "FAILED"
            except Exception as e:
                logger.error(f"ERR:{file_path.name} | {type(e).__name__} : {e}")
            return report

    async def save_to_json(self, file_path: Path, result: DialogueExtraction):
        output_file = Path(Config.OUTPUT_DIR) / f"{file_path.stem}_result.json"

        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(result.model_dump(), ensure_ascii=False, indent=4))
        logger.info(f"已完成：{file_path.name}->{output_file.name}")

    async def run(self):
        files = sorted(Config.INPUT_DIR.glob("*.txt"))
        if not files:
            logger.warning("No input files found.")
            return

        tasks = [self.extract_dialogue(file) for file in files]
        start_t = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_t
        success_count = sum(1 for r in results if r["status"] == "SUCCESS")
        logger.info(f"""
{"=" * 30}
Final Report:
Total time: {total_time:.2f}s
Success: {success_count}/{len(results)}
Average: {total_time / len(results):.2f}s/file
{"=" * 30}
""")


if __name__ == "__main__":
    extract = Extract()
    asyncio.run(extract.run())
