import os
from openai import OpenAI
import requests
from pathlib import Path
import re
from utils.logger import setup_logger
from utils.config import Config
from utils.config import ModelConfig

logger = setup_logger(__name__)


class Crawler:

    def __init__(self):
        self.client = OpenAI(api_key=ModelConfig.API_KEY,
                             base_url=ModelConfig.BASE_URL)
        self.chapter_prompt = self.load_prompt(Config.PROMPT_PATH /
                                               'chapter_url.md')
        self.crawler_prompt = self.load_prompt(Config.PROMPT_PATH /
                                               'crawler.md')
        self.save_chapter_path = Path.cwd() / 'chapter_url.txt'
        self.save_crawler_path = Path.cwd() / 'crawler.py'

    def load_prompt(self, prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()

    def save_file(self, file_path, content):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def get_html(self, url):
        html = requests.get(url)
        if html.status_code == 200:
            html.encoding = html.apparent_encoding
            return html.text

    def call_LLM(self, prompt, html):
        message = [{
            "role": "system",
            "content": prompt
        }, {
            "role": "user",
            "content": html
        }]
        response = self.client.chat.completions.create(
            model=ModelConfig.MODEL_NAME, messages=message)
        return response.choices[0].message.content

    def run(self, url):
        logger.info(f'正在爬取 {url}...')
        html = self.get_html(url)
        logger.info(f'获取成功！')
        logger.info(f'正在提取章节链接...')
        chapter_url = self.call_LLM(self.chapter_prompt, html)
        logger.info(f'提取成功！')
        logger.info(f'正在保存结果...')
        self.save_file(self.save_chapter_path, chapter_url)
        logger.info(f'保存成功！')
        first_line = chapter_url.split('\n')[0]
        pattern = r'\((https?://[^\)]+)\)'
        first_url = re.search(pattern, first_line).group(1)
        if first_url:
            logger.info(f'正在提取 {first_url}的内容...')
            first_html = self.get_html(first_url)
            logger.info(f'提取成功！')
            logger.info(f'正在生成爬虫代码...')
            crawler_code = self.call_LLM(self.crawler_prompt, first_html)
            if crawler_code:
                logger.info(f'生成成功！')
                if crawler_code[:3] == '```':
                    crawler_code = crawler_code[9:-3]
                logger.info(f'正在保存爬虫代码...')
                self.save_file(self.save_crawler_path, crawler_code)
                logger.info(f'保存成功！')


if __name__ == '__main__':
    crawler = Crawler()
    crawler.run()
