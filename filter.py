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

ZHI_PU_API_KEY = os.environ.get("ZHI_PU_API_KEY")


INPUT_DIR = Path("./提取结果")
OUTPUT_DIR = Path("./过滤结果")
LOG_DIR = Path("./filter_logs")
MAX_CONCURRENT_REQUESTS = 3
MAX_RETRIES = 3
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_filename = datetime.now().strftime("filter_%Y%m%d_%H%M%S.log")
log_path = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
)
system_prompt = """
# Role
你是一位精通《西游记》的资深文学评论家和数据标注专家。你的任务是从一堆原始台词中，筛选出能够鲜明体现“孙悟空”性格特征的高质量语料。

# Definition of "Sun Wukong" (性格维度)
请根据以下维度判断台词的含金量：
1. **桀骜与霸气**：体现齐天大圣的威严、自信、不畏强权（如：“皇帝轮流做，明年到我家”）。
2. **幽默与嘲讽**：对他人的调侃、机智的骂人话、特有的口癖（如：“呆子”、“甚至是嘲笑妖怪”）。
3. **情义与责任**：对唐僧的复杂情感、对猴子猴孙的护短、对取经重任的担当。
4. **标志性词汇**：包含“俺老孙”、“老孙”、“大圣”等强烈自我指涉的词汇。

# Filtering Rules (过滤规则)
1. **剔除平庸**：剔除所有通用的、换个角色也能说的功能性对话（如：“我们走吧”、“前面有座山”、“好的师父”）。
2. **剔除断章**：剔除缺乏上下文就无法理解的碎片短句。
3. **保留特色**：保留那些读起来就能脑补出孙悟空的句子。

# Input Format
我将提供一个包含多个句子的列表。

# Output Format
请返回一个 JSON 格式的列表，只包含符合要求的句子。并在 `reason` 字段简要说明符合哪个性格维度。

```json
[
    {{
        "original_text": "玉帝老儿，你若不依，俺老孙便把这凌霄宝殿搅个底朝天！",
        "reason": "桀骜与霸气，典型的大闹天宫风格"
    }},
    ...
]
"""


class FilteredItem(BaseModel):
    original_text: str = Field(..., description="原始台词")
    reason: str = Field(..., description="符合的性格维度")


class FilteredQuotes(BaseModel):
    selected_quotes: list[FilteredItem]


model = (
    ChatZhipuAI(
        model="glm-4.5-air",
        api_key=ZHI_PU_API_KEY,
        temperature=0.1,
        timeout=500,
    )
    .with_structured_output(FilteredQuotes)
    .with_retry(
        stop_after_attempt=MAX_RETRIES,
        wait_fixed=5,
        retry_if_exception_type=(
            Exception,
        ),  # 生产环境建议具体化，如 httpx.NetworkError
    )
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{text}")]
)

chain = prompt | model


async def process_single_file(file_path: Path, semaphore: asyncio.Semaphore):
    """处理单个文件的异步任务"""
    async with semaphore:
        status_report = {
            "filename": file_path.name,
            "status": "FAILED",
            "count": 0,
            "error": None,
            "duration": 0.0,
        }
        logging.info(f"开始处理文件:{file_path.name}")
        start_t = time.perf_counter()
        try:
            # 异步读取
            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                quotes_list = data.get("quotes", [])

            if not quotes_list:
                status_report["error"] = "文件内容为空"
                return status_report

            result: FilteredQuotes = await chain.ainvoke(
                {"text": json.dumps(quotes_list, ensure_ascii=False)}
            )
            # 异步写入
            output_file = OUTPUT_DIR / f"filtered_{file_path.name}"
            async with aiofiles.open(output_file, mode="w", encoding="utf-8") as f:
                await f.write(
                    json.dumps(result.model_dump(), ensure_ascii=False, indent=4)
                )
            status_report["status"] = "SUCCESS"
            status_report["count"] = len(result.selected_quotes)
            logging.info(
                f"√ 处理完成: {file_path.name} (保留: {status_report['count']})"
            )
        except Exception as e:
            # 捕获详细错误栈
            error_type = type(e).__name__
            status_report["error"] = f"[{error_type}] {str(e)}"
            logging.error(f"× 文件 {file_path.name} 处理失败: {status_report['error']}")
        finally:
            status_report["duration"] = round(time.perf_counter() - start_t, 2)
            if status_report["status"] == "SUCCESS":
                logging.info(
                    f"完成: {file_path.name} | 耗时: {status_report['duration']}s"
                )
            return status_report


async def main_async():
    if not os.path.exists(OUTPUT_DIR):
        logging.info(f"创建输出目录: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    input_path = Path(INPUT_DIR)
    files = sorted(list(input_path.glob("*.json")))

    if not files:
        logging.error(f"在{INPUT_DIR}下没有找到.json文件")
        return
    logging.info(
        f"任务启动 | 总文件数: {len(files)} | 并发数: {MAX_CONCURRENT_REQUESTS}"
    )
    global_start_t = time.perf_counter()
    # 创建信号量对象
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # 创建任务列表
    tasks = [process_single_file(file_path, semaphore) for file_path in files]
    # 执行任务并收集所有任务结果
    results = await asyncio.gather(*tasks)
    # --- 生成最终汇总报告 ---
    total_duration = round(time.perf_counter() - global_start_t, 2)
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    fail_count = len(results) - success_count
    total_quotes = sum(r["count"] for r in results)

    report_header = (
        "\n" + "=" * 50 + "\n" + "任务处理汇总报告".center(42) + "\n" + "=" * 50
    )
    logging.info(report_header)

    for r in results:
        indicator = " [√]" if r["status"] == "SUCCESS" else " [×]"
        msg = f"{indicator} {r['filename']:<20} | 状态: {r['status']:<8} | 数量: {r['count']:<3}"
        if r["error"]:
            msg += f" | 错误原因: {r['error']}"
        logging.info(msg)

    footer = (
        f"\n{'=' * 50}\n"
        f"统计总览:\n"
        f" - 总耗时: {total_duration}s\n"
        f" - 成功文件数: {success_count}\n"
        f" - 失败文件数: {fail_count}\n"
        f" - 提取台词总数: {total_quotes}\n"
        f"{'=' * 50}"
    )
    logging.info(footer)


if __name__ == "__main__":
    # 启动异步事件循环
    asyncio.run(main_async())
