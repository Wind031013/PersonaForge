import os
import asyncio
import aiofiles
import time
import json
import logging
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class Config:
    INPUT_DIR = Path("./data/过滤结果")
    OUTPUT_DIR = Path("./data/构筑结果")
    LOG_DIR = Path("./logs/Reconstruct_logs")
    MAX_CONCURRENT = 3
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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_DIR / log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

system_prompt = """
# Role
你是一位对话构筑专家，擅长从回复反向推导对话场景以及提问。
# Task
我会给你一段台词（Output）。请你反向构筑2个用户的话语（Input）。
# Constraints
1. **场景搭建**: Scene必须有逻辑性,且与Output有直接关联。
2. **因果关系**: Input必须是引出该台词的原因，Output必须是该原因导致的回答。
# Work Flow
1. 根据Output去推导出不同的2钟Scene说出这句话的场景
2. 然后，根据Scene构筑Input
# Example
**User Input**：“你这个没眼色的老头儿！你且睁开眼看看，你孙外公手里这根铁棒，可曾怕过谁？”
**Output**：
{{
    "scene":"取经途中，孙悟空遭遇一名不识其身份的妖怪挑衅。面对对方的轻视，孙悟空火眼金睛圆睁，正手持如意金箍棒，言语狂傲地进行反击。",
    "input":"你这瘦小的毛猴，也敢在此叫阵？趁早纳命来，免得我费力气！",
    "output":"你这个没眼色的老头儿！你且睁开眼看看，你孙外公手里这根铁棒，可曾怕过谁？"
}},
{{
    "scene":"孙悟空正欲降妖，土地或当地老者因畏惧妖怪神通，出言质疑金箍棒的威力。孙悟空闻言大怒，认为威严受损，故而以此狂言彰显神兵之威与自身底气。",
    "input":"大圣且慢，那妖怪法力无边，你这根铁条恐怕经受不住他的宝贝。",
    "output":"你这个没眼色的老头儿！你且睁开眼看看，你孙外公手里这根铁棒，可曾怕过谁？"
}},
"""


# --- 数据模型 ---
class Pairs(BaseModel):
    scene: str = Field(..., description="具体的对话场景描述")
    input: str = Field(..., description="用户的话语")
    output: str = Field(..., description="孙悟空的原始台词")


class BatchPairs(BaseModel):
    pairs: list[Pairs]


class Reconstructor:
    def __init__(self):
        self.model = ChatZhipuAI(
            model=Config.MODEL_NAME,
            api_key=Config.ZHI_PU_API_KEY,
            temperature=0.7,
            timeout=300,
        ).with_structured_output(BatchPairs)

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{text}")]
        )
        self.chain = self.prompt | self.model
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def call_llm(self, text: str) -> BatchPairs:
        """LLM调用"""
        return await self.chain.ainvoke({"text": text})

    async def process_batch(
        self, file_path: Path, batch_idx: int, quotes: list[str]
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
                    [f"台词{i + 1}:{q}" for i, q in enumerate(quotes)]
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

    async def save_to_jsonl(self, file_path: Path, pairs: list[Pairs]):
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

        results = await asyncio.gather(*tasks)

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
