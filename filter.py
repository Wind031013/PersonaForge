import os
import json
import logging
import asyncio
import time
import aiofiles
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)


# --- 配置 ---
class Config:
    INPUT_DIR = Path("./data/Extracted")
    OUTPUT_DIR = Path("./data/Filtered")
    LOG_DIR = Path("./logs/Filter_logs")
    ZHI_PU_API_KEY = os.environ.get("ZHI_PU_API_KEY")
    MODEL_NAME = "glm-4.5-air"
    MAX_CONCURRENT = 3

    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


Config.setup()

# --- 日志配置 ---
log_filename = datetime.now().strftime("filter_%Y%m%d_%H%M%S.log")
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
你是一位精通文学分析与人物刻画的资深评论家和数据标注专家。
你的任务是从原始台词中，筛选出能够鲜明体现“{target_role}”性格特征的高质量语料。

# Task
请帮我筛选出能够鲜明体现“{target_role}”性格特征的高质量语料以及这个句子所体现的性格或语气特征。

# Context
1. **保留**：具有独特语气（如：俺老孙、呆子）、强烈情感、或体现身份背景的句子。
2. **剔除**：通用废话（如“好的”、“你好”、“没问题”）、无上下文的碎片、以及不符合角色人设的句子。
"""


# --- 数据模型 ---
class FilteredItem(BaseModel):
    original_text: str = Field(..., description="原始台词")
    reason: str = Field(..., description="符合的性格维度")


class FilteredQuotes(BaseModel):
    quotes: list[FilteredItem] = Field(default_factory=list)


class Filter:
    def __init__(self, target_role: str):
        self.target_role = target_role
        self.model = ChatZhipuAI(
            model=Config.MODEL_NAME,
            api_key=Config.ZHI_PU_API_KEY,
            temperature=0.1,
            timeout=600,
        ).with_structured_output(FilteredQuotes)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "待筛选列表：\n{text}")]
        )
        self.chain = self.prompt | self.model
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def call_llm(self, text: str) -> FilteredQuotes:
        """LLM调用"""
        return await self.chain.ainvoke({"target_role": self.target_role, "text": text})

    async def filter_quotes(self, file_path: Path) -> dict:
        """过滤台词"""
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
                        raise ValueError("Empty file")
                    raw_data = json.loads(content)
                    quotes = raw_data.get("quotes", [])

                if not quotes:
                    logger.warning(f"文件{file_path.name}为空,跳过此文件")
                    report["status"] = "SKIPPED"
                    return report

                logger.info(f"正在过滤: {file_path.name} ({len(quotes)} 句)...")
                result: FilteredQuotes = await self.call_llm(
                    json.dumps(quotes, ensure_ascii=False)
                )
                if result and isinstance(result, FilteredQuotes):
                    await self.save_to_json(file_path, result)
                    report.update(
                        {
                            "status": "SUCCESS",
                            "count": len(result.quotes),
                            "duration": time.perf_counter() - start_t,
                        }
                    )
                    logger.info(
                        f"文件{file_path.name}过滤成功(保留: {report['count']})"
                    )
                else:
                    report["status"] = "FAILED"
                    logger.warning(f"文件{file_path.name}过滤失败")
                    return report
            except Exception as e:
                logger.error(f"ERR:{file_path.name} | {type(e).__name__} : {e}")

            return report

    async def save_to_json(self, file_path: Path, result: FilteredQuotes):
        output_path = Config.OUTPUT_DIR / f"{file_path.stem}_filtered.json"
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(result.model_dump(), ensure_ascii=False, indent=4))

    async def run(self):
        files = sorted(Config.INPUT_DIR.glob("*.json"))
        if not files:
            logger.warning("No input files found.")
            return
        tasks = [self.filter_quotes(file) for file in files]
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
    filter = Filter("西游记中的孙悟空")
    asyncio.run(filter.run())
