import requests
from bs4 import BeautifulSoup
import os
import time
import random
from urllib.parse import urljoin

class NovelDownloader:
    def __init__(self, start_url, output_dir):
        self.start_url = start_url
        self.output_dir = output_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://5000yan.com/'
        }
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_html(self, url):
        """带重试机制的请求函数"""
        for i in range(3):
            try:
                time.sleep(random.uniform(1.0, 2.0))  # 尊重的限速 
                resp = requests.get(url, headers=self.headers, timeout=15)
                resp.encoding = 'utf-8'
                if resp.status_code == 200:
                    return resp.text
            except Exception as e:
                print(f"请求异常 ({i+1}/3): {url}, 错误: {e}")
        return None

    def parse_and_save(self):
        current_url = self.start_url
        chapter_count = 1  # 用于命名文件：1.txt, 2.txt...

        while current_url:
            print(f"正在处理第 {chapter_count} 章: {current_url}")
            html = self.get_html(current_url)
            if not html:
                print("无法获取网页内容，停止。")
                break

            soup = BeautifulSoup(html, 'lxml')

            # 1. 提取标题 [cite: 13]
            title_tag = soup.find('h5', class_='text-center')
            title = title_tag.get_text(strip=True) if title_tag else f"第{chapter_count}章"

            # 2. 提取正文逻辑 
            # 网站正文分布在 class='grap' 的 div 及其后续同级 div 中
            content_parts = []
            
            # 提取主要正文块 
            main_grap = soup.find('div', class_='grap')
            if main_grap:
                # 移除内部的图片和无用标签 [cite: 14, 15]
                for tag in main_grap.find_all(['img', 'script', 'style']):
                    tag.decompose()
                content_parts.append(main_grap.get_text(separator='\n', strip=True))

            # 提取后续可能的段落 div（针对该站某些章节内容分散的情况）
            # 查找 h5 标题之后的所有 div，直到遇到翻页容器为止
            if main_grap:
                for sibling in main_grap.find_next_siblings('div'):
                    if 'container' in sibling.get('class', []) or 'pcfzsc' in sibling.get('class', []):
                        break
                    # 排除评论区和无关组件 [cite: 20, 28]
                    if not any(cls in sibling.get('class', []) for cls in ['wqy-pl', 'blog-footer']):
                        txt = sibling.get_text(strip=True)
                        if txt: content_parts.append(txt)

            full_content = "\n\n".join(content_parts)

            # 3. 实时保存为独立文件
            file_name = f"{chapter_count}.txt"
            file_path = os.path.join(self.output_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"{title}\n\n")
                f.write(full_content)
            
            print(f"成功保存: {file_name}")

            # 4. 精准寻找“下一篇”链接 
            next_url = None
            # 找到包含“下一篇：”文字的 div 容器 
            footer_divs = soup.find_all('div', class_='text-truncate')
            for div in footer_divs:
                if '下一篇：' in div.get_text():
                    a_tag = div.find('a')
                    if a_tag and a_tag.get('href'):
                        next_url = urljoin(current_url, a_tag['href'])
                        break
            
            # 更新循环变量
            if next_url:
                current_url = next_url
                chapter_count += 1
            else:
                print("全部章节采集完成。")
                current_url = None

if __name__ == "__main__":
    # 配置区
    ENTRY_URL = "https://xiyouji.5000yan.com/qsn/1249.html"  # 第一回 [cite: 13]
    TARGET_FOLDER = "西游记_章节分块"
    
    downloader = NovelDownloader(ENTRY_URL, TARGET_FOLDER)
    downloader.parse_and_save()