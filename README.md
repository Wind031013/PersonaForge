# PersonaForge

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-orange.svg)
![RAG](https://img.shields.io/badge/RAG-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**角色扮演模型构建流水线 - 从文本到智能角色**

[功能特色](#-功能特色) • [技术栈](#-技术栈) • [快速开始](#-快速开始) • [使用示例](#-使用示例) • [项目结构](#-项目结构)

</div>

---

## 📖 项目简介

PersonaForge 是一个全自动的角色扮演模型构建流水线，能够从任意小说、剧本或文本中提取角色特征，构建高质量的对话数据集和RAG知识库。通过多Agent协作系统，不仅可以生成适合模型微调的对话数据，还能创建角色的长期记忆系统，让AI真正学会"成为"那个角色。

### 🎯 核心功能

- **完整流水线**：从文本爬取到模型微调的一站式解决方案
- **智能特征提取**：深度分析角色性格、语言习惯和说话风格
- **RAG知识库**：构建角色的记忆系统，实现一致的角色扮演
- **高质量数据**：生成适合微调的对话对，提升模型角色扮演能力

### 🎯 解决的核心问题

- **数据稀缺**：缺乏具有鲜明角色特征的高质量对话训练数据
- **成本高昂**：人工标注和筛选对话数据费时费力
- **质量参差**：原始文本中混入大量无关内容和低质量对话
- **角色一致性**：现有模型难以维持长期稳定的角色特征
- **缺乏深度**：简单的对话模仿无法体现角色深层性格

---

## ✨ 功能特色

### 🔍 智能爬取系统
- 📖 支持多种小说源网站
- 🔄 带重试机制的健壮网络请求
- ⏱️ 智能限速，尊重目标网站
- 📁 智能文本清理，去除广告和无关内容

### 🤖 多Agent协作流水线
#### 1. 对话提取Agent
- 🎯 基于先进LLM模型（GLM-4.5-Air等）
- 🏷️ 智能识别角色及其所有别名
- ✨ 精确提取对话原文，自动排除旁白和描述

#### 2. 特征筛选Agent
- 🎭 多维度性格分析（可自定义维度）
- 🔍 识别角色标志性语言习惯和口癖
- 📊 基于角色的性格特征评分系统
- 🎯 筛选最具代表性的高质量对话

#### 3. 对话重构Agent
- 🎪 每句台词生成多个对话场景
- 💬 构建 (Scene, Input, Output) 完整结构
- 🚀 异步批量处理，支持高并发
- 🎨 保持角色语言风格的一致性

#### 4. RAG知识库构建Agent
- 🧠 构建角色背景故事知识库
- 📚 提取关键事件和关系
- 🔗 建立知识图谱，支持角色推理
- 💾 生成向量数据库，支持长期记忆

---

## 🛠️ 技术栈

| 类别 | 技术选型 | 说明 |
|------|----------|------|
| **语言** | Python 3.12+ | 现代 Python 版本，性能优异 |
| **LLM 框架** | LangChain | 大语言模型应用开发框架 |
| **AI 模型** | 智谱 AI GLM-4.5-Air / OpenAI GPT | 支持多种LLM模型，灵活配置 |
| **RAG 框架** | LangChain + ChromaDB / FAISS | 向量数据库和检索增强生成 |
| **数据验证** | Pydantic | 类型安全的数据验证和序列化 |
| **HTTP 客户端** | requests + httpx | 健壮的网络请求支持 |
| **HTML 解析** | BeautifulSoup4 + lxml | 高效的网页内容解析 |
| **异步处理** | asyncio + aiofiles | 高性能异步文件操作 |
| **重试机制** | tenacity | 优雅的错误重试处理 |
| **进度展示** | tqdm | 友好的进度条显示 |
| **向量嵌入** | sentence-transformers | 高质量文本向量表示 |
| **知识图谱** | networkx + pyvis | 角色关系可视化 |
| **配置管理** | pyyaml | 灵活的配置文件管理 |

### 核心依赖

```toml
# pyproject.toml
dependencies = [
    # 核心框架
    "langchain>=1.2.3",              # LLM 应用框架
    "langchain-community>=0.4.1",    # LangChain 社区版
    "langchain-openai>=1.1.7",      # OpenAI 集成
    "langchain-chroma>=0.1.0",       # ChromaDB 集成

    # AI 模型
    "openai>=2.15.0",                # OpenAI API
    "zhipuai>=2.0.0",                # 智谱 AI API

    # RAG 和向量数据库
    "chromadb>=0.6.0",               # 向量数据库
    "sentence-transformers>=3.0.0",  # 文本嵌入
    "faiss-cpu>=1.8.0",              # 向量检索

    # Web 和异步
    "httpx>=0.28.1",                 # HTTP 客户端
    "aiofiles>=25.1.0",              # 异步文件操作
    "beautifulsoup4>=4.12.0",       # HTML 解析
    "lxml>=4.9.0",                   # XML/HTML 解析器

    # 数据处理
    "pandas>=2.3.3",                 # 数据处理
    "numpy>=1.24.0",                 # 数值计算
    "networkx>=3.1.0",               # 知识图谱
    "pyvis>=0.3.0",                  # 图形可视化

    # 工具库
    "tenacity>=9.1.2",               # 重试机制
    "tqdm>=4.67.1",                  # 进度条
    "pyyaml>=6.0",                   # YAML 配置文件
    "pydantic>=2.0.0",               # 数据验证
]
```

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd sun-wukong-dataset

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
# 或使用 uv（推荐）
uv pip install -r requirements.txt
```

### 配置 API 密钥

在使用前，请确保已配置智谱 AI API 密钥：

```bash
# 在终端设置环境变量
export ZHIPUAI_API_KEY="your-api-key-here"

# 或者在代码中直接设置
import os
os.environ["ZHIPUAI_API_KEY"] = "your-api-key-here"
```

### 运行流程

项目采用模块化设计，可分步运行：

```python
# 1. 配置角色信息
# config.yaml
character:
  name: "孙悟空"
  aliases: ["石猴", "美猴王", "猴王", "大圣", "齐天大圣"]
  source: "西游记"

# 2. 爬取小说内容
from crawler import download_novel_content
await download_novel_content("config.yaml")

# 3. 提取角色对话
from extraction import extract_character_dialogues
dialogues = await extract_character_dialogues("config.yaml")

# 4. 筛选高质量对话
from filtering import filter_character_dialogues
filtered_dialogues = await filter_character_dialogues("config.yaml")

# 5. 构建对话数据集
from reconstruction import reconstruct_training_dataset
dataset = await reconstruct_training_dataset("config.yaml")

# 6. 创建RAG知识库
from rag import build_character_knowledge_base
kb = await build_character_knowledge_base("config.yaml")

# 7. 完整流水线（一键执行）
from personaforge import run_pipeline
await run_pipeline("config.yaml")
```

---

## 💡 使用示例

### 配置角色

```yaml
# config/harry_potter.yaml
character:
  name: "Harry Potter"
  aliases: ["The Boy Who Lived", "Harry"]
  source: "Harry Potter Series"
  personality_dimensions:
    - "brave"
    - "loyal"
    - "modest"
    - "determined"
  speaking_style:
    - "uses British slang"
    - "occasionally sarcastic"
    - "formal in magical contexts"
```

### 提取任意角色对话

```python
from extraction import extract_character_dialogues

# 从《红楼梦》中提取林黛玉对话
config = "config/lin_daiyu.yaml"
dialogues = await extract_character_dialogues(config)

# 输出格式示例
print(dialogues[0])
# {
#     "character": "林黛玉",
#     "alias": "颦颦",
#     "dialogue": "花谢花飞飞满天，红消香断有谁怜？",
#     "context": "葬花吟",
#     "emotional_tone": "melancholic"
# }
```

### 多维度性格过滤

```python
from filtering import filter_character_dialogues

# 基于自定义性格特征过滤
config = "config/shakespeare.yaml"
filtered = await filter_character_dialogues(config)

# 可配置的过滤维度：
# - 性格特征（勇敢、机智、忧郁等）
# - 语言标志（特定词汇、句式）
# - 情感基调
# - 上下文相关性
```

### 对话数据集构建

```python
from reconstruction import reconstruct_training_dataset

# 为微调生成对话数据集
config = "config/darth_vader.yaml"
dataset = await reconstruct_training_dataset(config)

# 输出格式（适合微调）：
# [
#     {
#         "scene": "Luke Skywalker asks about his mother",
#         "input": "告诉我关于我母亲的事",
#         "output": "你母亲的事迹早已成为传说...",
#         "character_style": "威严、神秘、带有机械声"
#     }
# ]
```

### RAG知识库构建

```python
from rag import build_character_knowledge_base
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# 构建角色知识库
config = "config/motoko_kusanagi.yaml"
kb = await build_character_knowledge_base(config)

# 知识库包含：
# - 角色背景故事
# - 重要事件时间线
# - 性格特征分析
# - 人际关系网络
# - 价值观和信念

# 使用RAG增强角色扮演
def ask_character(question, kb):
    relevant_memories = kb.search_memories(question)
    context = format_memories(relevant_memories)
    response = llm.generate_with_context(
        question=question,
        character_context=context,
        style指南=kb.style_guide
    )
    return response
```

---

## 📁 项目结构

```
personaforge/
├── 📄 main.py                     # 项目入口和CLI工具
├── 📄 pyproject.toml             # 项目配置
├── 📄 README.md                  # 项目文档
├── 📄 .gitignore                 # Git 忽略文件
│
├── 🔧 core/                      # 核心模块
│   ├── __init__.py
│   ├── pipeline.py              # 流水线编排
│   ├── config.py                # 配置管理
│   └── models.py                # 数据模型
│
├── 🕷️ crawlers/                  # 爬虫模块
│   ├── __init__.py
│   ├── base_crawler.py          # 爬虫基类
│   ├── web_novel_crawler.py     # 网络小说爬虫
│   └── text_cleaner.py          # 文本清理
│
├── 🤖 agents/                    # AI Agent模块
│   ├── __init__.py
│   ├── dialogue_extractor.py     # 对话提取Agent
│   ├── character_filter.py      # 角色筛选Agent
│   ├── dialogue_reconstructor.py # 对话重构Agent
│   └── rag_builder.py           # RAG知识库构建Agent
│
├── 🧠 rag/                       # RAG模块
│   ├── __init__.py
│   ├── knowledge_base.py        # 知识库核心
│   ├── vector_store.py          # 向量存储
│   └── memory_manager.py        # 记忆管理
│
├── 📊 datasets/                  # 数据集模块
│   ├── __init__.py
│   ├── dialogue_dataset.py       # 对话数据集
│   ├── formatter.py              # 数据格式化
│   └── validator.py             # 数据验证
│
├── ⚙️ configs/                   # 配置文件
│   ├── default.yaml             # 默认配置
│   ├── characters/              # 角色配置
│   │   ├── sun_wukong.yaml
│   │   ├── harry_potter.yaml
│   │   └── lin_daiyu.yaml
│   └── sources/                 # 数据源配置
│       ├── xiyouji.yaml
│       └── hongloumeng.yaml
│
└── 📚 examples/                  # 示例代码
    ├── basic_usage.py           # 基础使用
    ├── custom_character.py       # 自定义角色
    └── pipeline_demo.py         # 流水线演示
```

### 模块说明

| 模块 | 功能 | 特点 |
|------|------|------|
| **pipeline.py** | 流水线编排 | 协调各个Agent的工作流程 |
| **crawlers/** | 多源爬虫 | 支持各种小说网站和文本源 |
| **agents/** | 智能Agent | 专门处理角色相关任务的AI系统 |
| **rag/** | 记忆系统 | 构建和维护角色长期记忆 |
| **datasets/** | 数据处理 | 生成高质量的训练数据 |
| **configs/** | 灵活配置 | 易于扩展新角色和数据源 |

---

## 🚀 高级特性

### 多角色并行处理
```python
# 同时处理多个角色
from personaforge import run_multiple_pipelines

character_configs = [
    "configs/characters/sun_wukong.yaml",
    "configs/characters/zhu_bajie.yaml",
    "configs/characters/tangseng.yaml"
]

await run_multiple_pipelines(character_configs)
```

### 自定义Agent模板
```python
# 创建自定义Agent
from agents.base_agent import BaseAgent

class CustomFilterAgent(BaseAgent):
    async def process(self, dialogues):
        # 实现自定义过滤逻辑
        return filtered_dialogues

# 注册到流水线
pipeline.register_agent("custom_filter", CustomFilterAgent)
```

### 知识图谱可视化
```python
# 导出角色关系图
from rag import export_character_graph

graph = export_character_graph("configs/characters/journey_west.yaml")
graph.render("journey_west_relationship.html")
```

### 模型微调集成
```python
# 生成适配LoRA微调的数据格式
from datasets import format_for_lora

# JSON格式适合LoRA微调
lora_dataset = format_for_lora(
    dataset,
    model_type="llama",
    max_length=1024
)
lora_dataset.save("sun_wukong_lora.json")
```

---

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献

1. **Fork 项目**
2. **创建特性分支** (`git checkout -b feature/AmazingFeature`)
3. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
4. **推送分支** (`git push origin feature/AmazingFeature`)
5. **创建 Pull Request**

### 开发指南

- 遵循 PEP 8 编码规范
- 使用异步编程模式提高性能
- 添加适当的类型注解
- 编写清晰的文档字符串
- 为所有新功能添加测试用例

### 添加新角色

1. 在 `configs/characters/` 下创建角色配置文件
2. 测试对话提取质量
3. 调整性格维度参数
4. 分享您的配置并贡献给社区！

### 添加新数据源

1. 在 `crawlers/` 下实现新的爬虫类
2. 继承 `BaseCrawler` 并实现必要方法
3. 在配置文件中注册新数据源
4. 添加相应的测试用例

---

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## 🙏 致谢

- [智谱 AI](https://open.bigmodel.cn/) - 提供强大的 GLM 模型支持
- [LangChain](https://github.com/langchain-ai/langchain) - 优秀的 LLM 应用开发框架
- [ChromaDB](https://www.trychroma.com/) - 开源向量数据库
- sentence-transformers - 高质量的文本嵌入模型
- 所有创造了不朽经典形象的作者们

---

## 📖 更多资源

- [文档](https://personaforge.readthedocs.io/) - 完整文档
- [示例配置](configs/) - 各种角色配置示例
- [更新日志](CHANGELOG.md) - 版本更新记录
- [常见问题](FAQ.md) - 常见问题解答

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请考虑给个 star！**

[🐛 报告问题](https://github.com/your-username/personaforge/issues) • [💡 提出建议](https://github.com/your-username/personaforge/discussions)

[回到顶部](#-personaforge)

</div>