import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from utils.config import ModelConfig
from utils.config import Config
from utils.logger import setup_logger
from utils.tools import load_text, save_json, load_json, process_file, method
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.crawler_agent import Crawler

logger = setup_logger(__name__)


class Pipeline:

    def __init__(self):
        self.client = AsyncOpenAI(api_key=ModelConfig.API_KEY,
                                  base_url=ModelConfig.BASE_URL)
        self.extract_prompt = load_text(Config.PROMPT_PATH / "extract.md")
        self.filter_prompt = load_text(Config.PROMPT_PATH / "filter.md")
        self.refactor_prompt = load_text(Config.PROMPT_PATH / "refactor.md")
        self.semaphore = asyncio.Semaphore(Config.Semaphore_Num)

    async def crawler(self, url):
        logger.info("正在生成爬虫代码...")
        crawler = Crawler()
        crawler.run(url)
        logger.info("正在执行爬虫代码...")
        proc = await asyncio.create_subprocess_exec('python',
                                                    'crawler.py',
                                                    cwd=Path.cwd())

        await proc.wait()
        if proc.returncode != 0:
            logger.error("爬虫代码执行失败！")

    async def run(self, url, role, alias):
        # await self.crawler(url)
        if self.extract_prompt and role:
            self.extract_prompt = self.extract_prompt.replace("{role}", role)
        if self.extract_prompt and alias:
            self.extract_prompt = self.extract_prompt.replace(
                "{alias}", alias)
        logger.info("正在生成数据处理代码...")
        await process_file(self.client, self.extract_prompt, logger,
                           self.semaphore, method.extract)
        await process_file(self.client, self.filter_prompt, logger,
                           self.semaphore, method.filter)
        await process_file(self.client, self.refactor_prompt, logger,
                           self.semaphore, method.refactor)


if __name__ == '__main__':
    url = "https://www.guoxuemao.com/shu/xiyouji.html"
    role = "孙悟空"
    alias = "行者、齐天大圣"
    pipeline = Pipeline()
    asyncio.run(pipeline.run(url, role, alias))
