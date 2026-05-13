import re

text = "[第一回](https://www.guoxuemao.com/items/shu/14167a57a4e9578a0612cc7d0c801de0.html)"

pattern = r"\((https?://[^\)]+)\)"
match = re.search(pattern, text)

if match:
    url = match.group(1) # 取第一捕获组
    print(url) 