from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
# Cập nhật trong config.py
DATABASE_PATH = "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\database\\products.db"

VECTOR_DB_HOST = "localhost"
VECTOR_DB_PORT = "19530"

# OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
# OPENAI_API_BASE = os.getenv("OPENAI_BASE_URL")

LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANK_MODEL= "BAAI/bge-reranker-v2-m3"

COLLECTIONS = [
    "product_name",
    "product_info",
    "warranty",
    "technical",
    "feature",
    "content",
    "full_promotion"
]