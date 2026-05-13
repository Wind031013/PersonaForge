import os
from openai import OpenAI
import requests
from pathlib import Path
import argparse
import re
from logs.logger import setup_logger

class Config:
    API_KEY = os.environ.get('ZHI_PU_API_KEY')
    BASE_URL = os.environ.get('ZHI_PU_BASE_URL')
    PROMPT_DIR = Path(__file__).parent / 'prompts'
    MODEL = 'glm-4.7'

logger = setup_logger(__name__)


class Crawler:

    def __init__(self):
        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)
        self.chapter_prompt = self.load_prompt(Config.PROMPT_DIR /
                                               'chapter_url.md')
        self.crawler_prompt = self.load_prompt(Config.PROMPT_DIR /
                                               'crawler.md')
        self.save_chapter_path = Path(__file__).parent / 'chapter_url.txt'
        self.save_crawler_path = Path(__file__).parent / 'crawler.py'

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
        response = self.client.chat.completions.create(model=Config.MODEL,
                                                       messages=message)
        return response.choices[0].message.content

    def run(self):
        parser = argparse.ArgumentParser(description="智能爬虫")
        parser.add_argument('-u', '--url', help='要爬取的网址')
        args = parser.parse_args()
        url = args.url
        url = "https://www.guoxuemao.com/shu/xiyouji.html"
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
                logger.info(f'正在保存爬虫代码...')
                self.save_file(self.save_crawler_path, crawler_code)
                logger.info(f'保存成功！')



if __name__ == '__main__':
    crawler = Crawler()
    crawler.run()
