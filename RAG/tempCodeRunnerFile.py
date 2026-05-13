import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb


class Config:
    DB_PATH = Path("./data_db")
    DATA_DIR_PATH = Path(__file__).resolve().parent.parent / "data" / "西游记"
    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class RagConstruction:

    def __init__(self, batch_size: int = 32):
        self.file_count, self.file_names = self.get_file_count()
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL, device="cuda")
        self.batch_size = batch_size
        self.data = self.load_data()
        self.chroma_client = chromadb.PersistentClient(path=Config.DB_PATH)
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
            print(f"Error: {e}")
            return None

    def load_data(self):
        data = []
        try:
            for file_name in self.file_names:
                file_path = Config.DATA_DIR_PATH / file_name
                file_content = self.file_split(file_path)
                data.append(file_content)
            return data
        except Exception as e:
            print(f"Error: {e}")
            return []

    def save_embeddings(self):
        if not self.data:
            print("No data to save.")
            return
        print(f"Saving embeddings...{self.file_count}files")
        try:
            print("Encoding embeddings...")
            embeddings = self.model.encode(
                self.data,
                batch_size=self.batch_size,
                show_progress_bar=True)
            embeddings_list = embeddings.tolist()
            print("Saving to ChromaDB...")
            ids = [f"chunk_{i}" for i in range(len(self.data))]
            self.collection.upsert(
                documents=self.data,
                ids=ids,
                embeddings=embeddings_list
            )
            print("Embeddings saved successfully.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    rag = RagConstruction()
    rag.save_embeddings()
