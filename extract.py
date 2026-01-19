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

ZHI_PU_API_KEY = os.environ.get("ZHI_PU_API_KEY")
INPUT_DIR = Path("./西游记_章节分块")
OUTPUT_DIR = Path("./提取结果")
MAX_WORKERS = 3
MAX_RETRIES = 3
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_filename = datetime.now().strftime("extract_%Y%m%d_%H%M%S.log")
log_path = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
)

system_prompt = """
# Role
你是一个精通中文文学的文本提取助手。

# Task
提取文档中[孙悟空]（及其别名：石猴、美猴王、猴王、大圣）的所有原话对话。

# Rules
1. 仅提取引号内的对话原文。
2. 排除旁白（例如：不要包含“那猴王道：”这些字）。
"""


class DialogueExtraction(BaseModel):
    quotes: list[str] = Field(description="该角色说的具体话语列表")


model = (
    ChatZhipuAI(
        model="glm-4.5-air",
        api_key=ZHI_PU_API_KEY,
        temperature=0.1,
        timeout=300,
    )
    .with_structured_output(DialogueExtraction)
    .with_retry(
        stop_after_attempt=MAX_RETRIES,
        retry_if_exception_type=(
            Exception,
        ),  # 生产环境建议具体化，如 httpx.NetworkError
    )
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{text}")]
)

chain = prompt | model


async def extract_dialogue(file_path: Path, semaphore: asyncio.Semaphore):
    """处理单个文件的异步任务"""
    async with semaphore:
        status_report = {
            "file_name": file_path.name,
            "status": "FAILED",
            "msg": "",
            "duration": 0.0,
        }
        logging.info(f"开始处理文件: {file_path.name}")
        start_t = time.perf_counter()
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            if not content.strip():
                logging.warning(f"文件{file_path.name}为空,跳过此文件")
                status_report["status"] = "SKIPPED"
                status_report["msg"] = "内容为空"
                return status_report
            logging.info(f"正在处理: {file_path.name} ({len(content)} 字)...")
            result: DialogueExtraction = await chain.ainvoke({"text": content})
            output_file = Path(OUTPUT_DIR) / f"{file_path.stem}_result.json"

            async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                await f.write(
                    json.dumps(result.model_dump(), ensure_ascii=False, indent=4)
                )
            logging.info(f"已完成：{file_path.name}->{output_file.name}")
            status_report["status"] = "SUCCESS"
            status_report["msg"] = "处理成功"
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")
            status_report["msg"] = str(e)
        finally:
            status_report["duration"] = round(time.perf_counter() - start_t, 2)
            if status_report["status"] == "SUCCESS":
                logging.info(
                    f"完成: {file_path.name} | 耗时: {status_report['duration']}s"
                )

            return status_report


async def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    input_path = Path(INPUT_DIR)
    files = sorted(list(input_path.glob("*.txt")))

    if not files:
        logging.error(f"在{INPUT_DIR}下没有找到.txt文件")
        return

    logging.info(f"任务启动 | 总文件数: {len(files)} | 并发数: {MAX_WORKERS}")
    global_start_t = time.perf_counter()
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    tasks = [extract_dialogue(file_path, semaphore) for file_path in files]

    results = await asyncio.gather(*tasks)
    report_header = (
        "\n" + "=" * 50 + "\n" + "任务处理汇总报告".center(42) + "\n" + "=" * 50
    )
    logging.info(report_header)
    total_duration = round(time.perf_counter() - global_start_t, 2)
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    fail_count = len(results) - success_count
    footer = (
        f"\n{'=' * 50}\n"
        f"统计总览：\n"
        f"总耗时: {total_duration}s\n"
        f"成功文件数:{success_count}\n"
        f"失败文件数：{fail_count}\n"
        f"{'=' * 50}"
    )
    logging.info(footer)
    for res in results:
        logging.info(res)


if __name__ == "__main__":
    asyncio.run(main())
