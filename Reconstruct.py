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
    INPUT_DIR = Path("./data/Filtered")
    OUTPUT_DIR = Path("./data/Reconstructed")
    LOG_DIR = Path("./logs/Reconstruct_logs")
    MAX_CONCURRENT = 2
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
你是一位金牌编剧和对话构筑专家，擅长从一句台词（Output）反向推导其背后的情感冲突、人物关系及特定场景。
# Task
我会给你一批台词（Output）。请你通过反向推理，分别给每一个台词构筑 2 个极具真实感、能够自然引出这句台词的对话数据。
# Constraints
1. **场景多样性（Scene）**: 两个场景必须有明显的差异。例如：场景一可以是“冲突对峙”，场景二可以是“日常请教等”。
2. **逻辑契合（Input -> Output）**: Input 必须包含引爆点（Trigger）。这个引爆点可以是对方的质疑、挑衅、请求或误解，从而迫使角色说出那句 Output。
3. **口语化与情绪化**: Input 的语言风格要符合场景设定，避免死板的叙述。如果 Output 情绪激烈，Input 也应具备相应的情绪张力。
# Work Flow
1. **情感分析**: 分析 Output 的情绪（狂傲、谦卑、兴奋、愤怒还是执着）。
2. **场景构筑**: 根据 Output 的情绪，构筑两个具有明显差异的场景。
3. **话语构筑**: 编写能够自然闭环的 Input。
# Example
**User Input**: “俺老孙上天入地，无所不能，怎么去不得！”
**Output**: 
{{"scene": "取经团队来到一处瘴气弥漫的绝地，猪八戒因畏难情绪在一旁冷嘲热讽，质疑孙悟空也没法带大家过去。", "input": "猴哥，这地方连神仙都要绕道，你那点本事怕是也悬，咱们还是散伙分行李吧！", "output": "俺老孙上天入地，无所不能，怎么去不得！"}}
{{"scene": "孙悟空前往禁地求药，守护灵兽见他并无仙职，阻拦其入内并出言轻视。", "input": "此乃神圣之地，非大罗金仙不得入内。你这野猴，莫要白白送了性命，快快退去！", "output": "俺老孙上天入地，无所不能，怎么去不得！"}}
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
                "status": "FAILED",
                "count": 0,
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
                    for item in data.get("quotes", [])
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
