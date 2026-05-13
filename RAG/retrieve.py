from sentence_transformers import SentenceTransformer
from pathlib import Path
import chromadb

class Config:
    DB_PATH = Path("./paragraph_data_db")
    DATA_DIR_PATH = Path(__file__).resolve().parent.parent / "data" / "西游记"
    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL, device="cuda")
        self.db_client = chromadb.PersistentClient(path=Config.DB_PATH)
        self.db_collection = self.db_client.get_or_create_collection(name="my_collection")

    def retrieve(self, query: str, top_k: int = 30) -> list[str]:
        query_embedding = self.model.encode(query)
        results = self.db_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        return results['documents'][0]

if __name__ == "__main__":
    retriever = Retriever()
    query = "紫金红葫芦是怎么收人的？"
    results = retriever.retrieve(query)
    for result in results:
        print(result)
        print("-" * 20)

        