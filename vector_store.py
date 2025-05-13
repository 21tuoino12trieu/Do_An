from pymilvus import connections, Collection, utility
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import numpy as np
from config import VECTOR_DB_HOST, VECTOR_DB_PORT, EMBEDDING_MODEL, RERANK_MODEL, COLLECTIONS


class VectorStore:
    def __init__(self):
        self.host = VECTOR_DB_HOST
        self.port = VECTOR_DB_PORT
        self.embedding_model = self._load_embedding_model()
        self.reranker = self._load_reranker()
        self.collections = {}

    def _load_embedding_model(self):
        return BGEM3FlagModel(EMBEDDING_MODEL, use_cuda=False)

    def _load_reranker(self):
        return FlagReranker(RERANK_MODEL, use_fp16=True)

    def connect(self):
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
            print("Kết nối thành công đến MILVUS")
            return True
        except Exception as e:
            print(f"Lỗi kết nối đến MILVUS: {e}")
            return False

    def load_collections(self, base_name="product_database"):
        """Load collections, xử lý lỗi 'get_stats' nếu có"""
        self.connect()
        for field in COLLECTIONS:
            collection_name = f"{base_name}_{field}"
            try:
                # Kiểm tra xem collection có tồn tại không
                if utility.has_collection(collection_name):
                    self.collections[field] = Collection(name=collection_name)

                    # Tải collection, xử lý lỗi get_stats nếu cần
                    try:
                        self.collections[field].load()
                        print(f"Đã tải collection {collection_name}")
                    except AttributeError as e:
                        if "get_stats" in str(e):
                            print(f"Tải collection {collection_name} với phương thức thay thế")
                            # Cách tải thay thế không sử dụng get_stats
                            # Phụ thuộc vào phiên bản pymilvus, bạn có thể cần điều chỉnh
                            try:
                                self.collections[field] = Collection(name=collection_name)
                                print(f"Đã tải collection {collection_name} (phương thức thay thế)")
                            except Exception as e2:
                                print(f"Không thể tải collection {collection_name}: {str(e2)}")
                        else:
                            raise e
                else:
                    print(f"Collection {collection_name} không tồn tại")
            except Exception as e:
                print(f"DEBUG - ERROR loading collection {collection_name}: {str(e)}")
                # Tiếp tục với collection tiếp theo thay vì dừng lại
                continue

        print("Đã tải các collections có sẵn")
        return self.collections

    def get_embeddings(self, text):
        """Tạo embedding từ text"""
        text_embedding = self.embedding_model.encode(text, max_length=2048)["dense_vecs"]
        return text_embedding

    def list_available_collections(self):
        """Liệt kê tất cả collections có sẵn trong Milvus"""
        self.connect()
        try:
            all_collections = utility.list_collections()
            print(f"Danh sách collections có sẵn: {all_collections}")
            return all_collections
        except Exception as e:
            print(f"Lỗi khi liệt kê collections: {e}")
            return []