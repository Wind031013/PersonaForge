
import os
import re
import time
import requests
from bs4 import BeautifulSoup

def get_novel_content():
    # 创建保存数据的文件夹
    if not os.path.exists('data'):
        os.makedirs('data')

    # 读取章节链接文件
    chapters = []
    with open('chapter_url.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 使用正则匹配 [章节名](链接) 格式
            pattern = re.compile(r'\[(.*?)\]\((.*?)\)')
            match = pattern.search(line)
            if match:
                title = match.group(1)
                url = match.group(2)
                chapters.append((title, url))

    # 设置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # 遍历所有章节
    for title, url in chapters:
        try:
            print(f"正在下载: {title} - {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 根据HTML结构解析正文内容
                # 正文在 div.book_content 下的 div.content 中
                content_div = soup.select_one('.book_content .content')
                
                if content_div:
                    paragraphs = content_div.find_all('p')
                    full_text = ""
                    for p in paragraphs:
                        # 提取段落文本并保留换行
                        text = p.get_text()
                        full_text += text + "\n\n"
                    
                    # 构造文件名
                    filename = os.path.join('data', f"{title}.txt")
                    
                    # 保存文件
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(full_text.strip())
                    print(f"保存成功: {filename}")
                else:
                    print(f"未找到正文内容: {url}")
            else:
                print(f"请求失败，状态码: {response.status_code}")
            
            # 延时，避免请求过快
            time.sleep(1)
            
        except Exception as e:
            print(f"下载出错: {title}, 错误信息: {e}")

if __name__ == '__main__':
    get_novel_content()