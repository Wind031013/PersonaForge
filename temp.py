import os
import json
import logging
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
from pydantic import BaseModel, Field, ValidationError
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)


# --- 配置管理 (Pydantic Style) ---
class Settings:
    INPUT_DIR = Path("./data/Extracted")
    OUTPUT_DIR = Path("./data/Filtered")
    LOG_DIR = Path("./logs/Filter_logs")
    ZHI_PU_API_KEY = os.environ.get("ZHI_PU_API_KEY")
    MODEL_NAME = "glm-4.6v-flashx"
    MAX_CONCURRENT = 3
    RETRY_ATTEMPTS = 3

    @classmethod
    def initialize(cls):
        for path in [cls.OUTPUT_DIR, cls.LOG_DIR]:
            path.mkdir(parents=True, exist_ok=True)


Settings.initialize()

# --- 日志系统 ---
log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(
            Settings.LOG_DIR / datetime.now().strftime("filter_%Y%m%d.log"),
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# --- 数据模型 ---
class FilteredItem(BaseModel):
    original_text: str = Field(..., description="符合筛选条件的原始台词文本")
    reason: str = Field(..., description="简短说明该台词体现的性格维度或语气特征")


class FilteredQuotes(BaseModel):
    quotes: List[FilteredItem] = Field(
        default_factory=list, description="筛选后的台词列表"
    )


# --- 提示词工程优化 ---
SYSTEM_PROMPT = """你是一位资深文学评论家。任务：从提供的对话列表中，筛选出能鲜明体现“{target_role}”性格特征的高质量语料。

# Task
1. **保留**：具有独特语气（如：俺老孙、呆子）、强烈情感、或体现身份背景的句子。
2. **剔除**：通用废话（如“好的”、“你好”、“没问题”）、无上下文的碎片、以及不符合角色人设的句子。

### 输出要求：
必须返回有效的 JSON 对象，格式为：{{"quotes": [ {{"original_text": "...", "reason": "..."}} ]}}
"""


class RoleQuoteFilter:
    def __init__(self, target_role: str):
        self.target_role = target_role
        # 初始化模型并绑定结构化输出
        llm = ChatZhipuAI(
            model=Settings.MODEL_NAME,
            api_key=Settings.ZHI_PU_API_KEY,
            temperature=0.1,
            timeout=100,
        )
        self.chain = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "待筛选列表：\n{text}")]
        ) | llm.with_structured_output(FilteredQuotes)

        self.semaphore = asyncio.Semaphore(Settings.MAX_CONCURRENT)

    @retry(
        stop=stop_after_attempt(Settings.RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        retry=retry_if_exception_type((Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=False,  # 关键：耗尽重试后不抛出异常，由函数内部处理
    )
    async def _call_llm_core(self, text_input: str) -> Optional[FilteredQuotes]:
        """核心调用逻辑，处理模型返回的结构化数据"""
        return await self.chain.ainvoke(
            {"target_role": self.target_role, "text": text_input}
        )

    async def process_file(self, file_path: Path) -> dict:
        """单文件处理逻辑"""
        async with self.semaphore:
            stats = {"file": file_path.name, "status": "INIT", "saved": 0}
            try:
                # 1. 安全读取
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    if not content.strip():
                        raise ValueError("Empty file")
                    raw_data = json.loads(content)
                    quotes = raw_data.get("quotes", [])

                if not quotes:
                    stats["status"] = "SKIPPED"
                    return stats

                # 2. 调用 LLM
                logger.info(f"Processing {file_path.name} ({len(quotes)} quotes)...")
                # 将列表转为带编号的文本，有助于 LLM 定位
                formatted_input = "\n".join([f"- {q}" for q in quotes])

                result = await self._call_llm_core(formatted_input)

                # 3. 结果校验与保存
                if result and isinstance(result, FilteredQuotes):
                    await self._save_result(file_path, result)
                    stats.update({"status": "SUCCESS", "saved": len(result.quotes)})
                else:
                    stats["status"] = "LLM_ERROR"
                    logger.error(
                        f"Failed to get valid structured output for {file_path.name}"
                    )

            except json.JSONDecodeError:
                stats["status"] = "JSON_CORRUPT"
                logger.error(f"Invalid JSON format in {file_path.name}")
            except Exception as e:
                stats["status"] = "ERROR"
                logger.error(f"Unexpected error processing {file_path.name}: {str(e)}")

            return stats

    async def _save_result(self, original_path: Path, data: FilteredQuotes):
        output_path = Settings.OUTPUT_DIR / f"{original_path.stem}_filtered.json"
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(data.model_dump_json(indent=4, build_as_node=True))

    async def run(self):
        input_files = sorted(Settings.INPUT_DIR.glob("*.json"))
        if not input_files:
            logger.warning("No files found in input directory.")
            return

        start_time = time.perf_counter()
        tasks = [self.process_file(f) for f in input_files]
        results = await asyncio.gather(*tasks)

        # 统计
        duration = time.perf_counter() - start_time
        success_files = [r for r in results if r["status"] == "SUCCESS"]

        logger.info(f"""
{"=" * 40}
PROCESSING COMPLETE
Total Files: {len(input_files)}
Success: {len(success_files)}
Failed/Skipped: {len(input_files) - len(success_files)}
Total Time: {duration:.2f}s
Avg Speed: {duration / len(input_files):.2f}s/file
{"=" * 40}
""")


if __name__ == "__main__":
    # 使用更具体的角色描述有助于 LLM 过滤
    processor = RoleQuoteFilter("《西游记》中的孙悟空（齐天大圣）")
    asyncio.run(processor.run())
