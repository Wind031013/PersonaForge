import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import re


class Config:
    FILE_DB_PATH = Path("./file_data_db")
    FIXED_DB_PATH = Path("./fixed_data_db")
    PARAGRAPH_DB_PATH = Path("./paragraph_data_db")
    DATA_DIR_PATH = Path(__file__).resolve().parent.parent / "data" / "西游记"
    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50


class RagConstruction:

    def __init__(self, split_mode: str = "file", batch_size: int = 32):
        """
        split_mode: 文档切割模式,可选: file(按文件), fixed(固定长度), paragraph(按段落)
        batch_size: 批量处理文件大小
        """
        self.split_mode = split_mode
        self.file_count, self.file_names = self.get_file_count()
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL, device="cuda")
        self.batch_size = batch_size

        db_path_map = {
            "file": Config.FILE_DB_PATH,
            "fixed": Config.FIXED_DB_PATH,
            "paragraph": Config.PARAGRAPH_DB_PATH
        }
        self.db_path = db_path_map.get(split_mode, Config.FILE_DB_PATH)

        # 初始化客户端和集合，集合名也可以加上模式后缀区分
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.chroma_client.get_or_create_collection(
            name="my_collection")

    def get_file_count(self, data_path: Path = Config.DATA_DIR_PATH):
        try:
            count = 0
            file_names = []
            with os.scandir(data_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        count += 1
                        file_names.append(entry.name)
            return count, file_names
        except Exception as e:
            print(f"Error: {e}")
            return 0, []

    def file_split(self, file_path: Path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def fixed_length_split(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """固定字符长度分隔，带重叠"""
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            chunks.append(text[start:end])
            # 如果剩余文本不足chunk_size，直接结束
            if end >= text_len:
                break
            # 移动起点，加上重叠长度
            start += chunk_size - overlap
        return chunks

    def paragraph_split(self, text: str) -> list[str]:
        """段落分隔（支持换行符、连续换行、中文缩进等智能清洗）"""
        # 先按单换行或连续换行进行初步分割
        raw_chunks = re.split(r'\n+', text)
        cleaned_chunks = []
        for chunk in raw_chunks:
            # 去除首尾空格，如果是空行直接跳过
            chunk = chunk.strip()
            if not chunk:
                continue
            # 处理中文小说常见的缩进（去除掉如“　　”或“    ”开头的空格）
            chunk = re.sub(r'^[\s\u3000]+', '', chunk)
            if chunk:
                cleaned_chunks.append(chunk)
        return cleaned_chunks

    def load_data(self):
        """
        统一的数据加载与切片入口
        返回格式: list[dict]，每个dict包含 text, metadata(文件名, 切片索引)
        """
        data = []
        try:
            for file_idx, file_name in enumerate(self.file_names):
                file_path = Config.DATA_DIR_PATH / file_name
                file_content = self.file_split(file_path)

                if not file_content:
                    continue

                # 1. 按原文件分隔（兼容旧逻辑）
                if self.split_mode == "file":
                    chunks = [file_content]

                # 2. 固定字符分隔
                elif self.split_mode == "fixed":
                    chunks = self.fixed_length_split(
                        file_content,
                        Config.CHUNK_SIZE,
                        Config.CHUNK_OVERLAP
                    )

                # 3. 按段落分隔
                elif self.split_mode == "paragraph":
                    chunks = self.paragraph_split(file_content)
                else:
                    chunks = [file_content]

                # 将切片结果组装成标准格式
                for chunk_idx, chunk_text in enumerate(chunks):
                    data.append({
                        "text": chunk_text,
                        "metadata": {
                            "source_file": file_name,
                            "chunk_index": chunk_idx
                        }
                    })
            return data
        except Exception as e:
            print(f"Error in load_data: {e}")
            return []

    def save_embeddings(self):
        if not self.data:
            print("No data to save.")
            return

        print(f"[{self.split_mode.upper()} MODE] Saving embeddings for {len(self.data)} chunks from {self.file_count} files...")
        try:
            # 提取纯文本用于 Encode
            texts = [item["text"] for item in self.data]

            print("Encoding embeddings...")
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True
            )
            embeddings_list = embeddings.tolist()

            metadatas = [item["metadata"] for item in self.data]

            ids = [f"{item['metadata']['source_file']}_{item['metadata']['chunk_index']}" for item in self.data]

            print(f"Saving to ChromaDB at: {self.db_path} ...")
            self.collection.upsert(
                documents=texts,
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas  # 强烈建议带上 metadata
            )
            print("Embeddings saved successfully.")
        except Exception as e:
            print(f"Error in save_embeddings: {e}")


if __name__ == "__main__":

    print("--- 测试固定字符切片 ---")
    rag_fixed = RagConstruction(split_mode="fixed", batch_size=16)
    rag_fixed.data = rag_fixed.load_data()
    rag_fixed.save_embeddings()

    print("\n--- 测试段落切片 ---")
    rag_para = RagConstruction(split_mode="paragraph", batch_size=16)
    rag_para.data = rag_para.load_data()
    rag_para.save_embeddings()

    print("\n--- 测试文件切片 ---")
    rag_file = RagConstruction(split_mode="file", batch_size=8)
    rag_file.data = rag_file.load_data()
    rag_file.save_embeddings()
