import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ConfigDict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


# --- 配置中心 ---
class Config:
    INPUT_DIR = Path("./过滤结果")
    OUTPUT_DIR = Path("./data/构筑结果")
    LOG_DIR = Path("./Reconstruct_logs")
    MAX_CONCURRENT = 5  # 适度增加并发
    BATCH_SIZE = 5
    MODEL_NAME = "glm-4.5-air"
    ZHI_PU_API_KEY = os.environ.get("ZHI_PU_API_KEY")

    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


Config.setup()

# --- 日志配置 ---
log_filename = datetime.now().strftime("Reconstruct_%Y%m%d_%H%M%S.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_DIR / log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("Reconstructor")


# --- 数据模型 ---
class Pairs(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    scene: str = Field(..., description="具体的对话场景描述")
    input: str = Field(..., description="用户的话语")
    output: str = Field(..., description="孙悟空的原始台词")


class BatchPairs(BaseModel):
    pairs: List[Pairs]


# --- 系统提示词优化 ---
SYSTEM_PROMPT = """你是一位精通《西游记》的对话专家。
任务：根据给出的孙悟空单向台词（Output），反向构筑2个逻辑严密的对话场景。
要求：
1. 严格返回JSON格式。
2. 每个Output必须对应2个不同的Scene和Input。
3. 确保 scene, input, output 三个字段完整，不要合并字段。
"""


# --- 核心逻辑类 ---
class Reconstructor:
    def __init__(self):
        self.model = ChatZhipuAI(
            model=Config.MODEL_NAME,
            api_key=Config.ZHI_PU_API_KEY,
            temperature=0.7,
            timeout=120,
        ).with_structured_output(BatchPairs)

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "请根据以下台词构筑对话：\n{text}")]
        )
        self.chain = self.prompt | self.model
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def call_llm(self, combined_text: str) -> BatchPairs:
        """带重试机制的LLM调用"""
        return await self.chain.ainvoke({"text": combined_text})

    async def process_batch(
        self, file_path: Path, batch_idx: int, quotes: List[str]
    ) -> dict:
        async with self.semaphore:
            start_t = time.perf_counter()
            report = {
                "filename": file_path.name,
                "count": 0,
                "status": "FAILED",
                "duration": 0,
            }

            try:
                combined_input = "\n".join(
                    [f"台词{i + 1}: {q}" for i, q in enumerate(quotes)]
                )
                result = await self.call_llm(combined_input)

                if result and result.pairs:
                    await self.save_to_jsonl(file_path, result.pairs)
                    report.update(
                        {
                            "status": "SUCCESS",
                            "count": len(result.pairs),
                            "duration": time.perf_counter() - start_t,
                        }
                    )
                    logger.info(
                        f"OK: {file_path.name} Batch {batch_idx + 1} (+{len(result.pairs)})"
                    )
            except Exception as e:
                logger.error(
                    f"ERR: {file_path.name} Batch {batch_idx + 1} | {type(e).__name__}: {e}"
                )

            return report

    async def save_to_jsonl(self, file_path: Path, pairs: List[Pairs]):
        output_path = Config.OUTPUT_DIR / f"{file_path.stem}_reconstructed.jsonl"
        async with aiofiles.open(output_path, "a", encoding="utf-8") as f:
            lines = [json.dumps(p.model_dump(), ensure_ascii=False) for p in pairs]
            await f.write("\n".join(lines) + "\n")

    async def run(self):
        files = sorted(list(Config.INPUT_DIR.glob("*.json")))
        if not files:
            logger.warning("No input files found.")
            return

        tasks = []
        for file_path in files:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
                quotes = [
                    item["original_text"]
                    for item in data.get("selected_quotes", [])
                    if item.get("original_text")
                ]

                for i in range(0, len(quotes), Config.BATCH_SIZE):
                    tasks.append(
                        self.process_batch(
                            file_path,
                            i // Config.BATCH_SIZE,
                            quotes[i : i + Config.BATCH_SIZE],
                        )
                    )

        logger.info(f"Task Started: {len(files)} files, {len(tasks)} batches.")
        start_time = time.perf_counter()

        # 使用 gather 获取所有结果用于最后统计
        results = await asyncio.gather(*tasks)

        # 统计分析
        total_time = time.perf_counter() - start_time
        success_pairs = sum(r["count"] for r in results if r["status"] == "SUCCESS")
        logger.info(f"""
{"=" * 30}
Final Report:
- Total Time: {total_time:.2f}s
- Success Pairs: {success_pairs}
- Avg Throughput: {success_pairs / total_time:.2f} pairs/s
{"=" * 30}
""")


if __name__ == "__main__":
    reconstructor = Reconstructor()
    asyncio.run(reconstructor.run())
