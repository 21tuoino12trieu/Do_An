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
        self.connection_established = False

    def _load_embedding_model(self):
        try:
            return BGEM3FlagModel(EMBEDDING_MODEL, use_cuda=False)
        except Exception as e:
            print(f"Lỗi khi tải embedding model: {e}")
            return None

    def _load_reranker(self):
        try:
            return FlagReranker(RERANK_MODEL, use_fp16=True)
        except Exception as e:
            print(f"Lỗi khi tải reranker: {e}")
            return None

    def connect(self):
        """Kết nối đến Milvus và đảm bảo kết nối tồn tại"""
        try:
            # Kiểm tra xem kết nối đã tồn tại hay chưa
            try:
                # Thử thực hiện một hoạt động để kiểm tra kết nối
                utility.list_collections()
                self.connection_established = True
                return True
            except Exception:
                # Kết nối không tồn tại, tạo kết nối mới
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                )
                print("Kết nối thành công đến MILVUS")
                self.connection_established = True
                return True
        except Exception as e:
            print(f"Lỗi kết nối đến MILVUS: {e}")
            self.connection_established = False
            return False

    def disconnect(self):
        """Ngắt kết nối đến Milvus"""
        try:
            connections.disconnect("default")
            self.connection_established = False
            print("Đã ngắt kết nối MILVUS")
            return True
        except Exception as e:
            print(f"Lỗi khi ngắt kết nối MILVUS: {e}")
            return False

    def load_collections(self, base_name="product_database"):
        """Load collections, xử lý lỗi 'get_stats' nếu có"""
        # Đảm bảo kết nối được thiết lập
        if not self.connection_established:
            self.connect()

        # Nếu đã load collections trước đó, return để tránh load lại
        if self.collections and len(self.collections) > 0:
            print("Các collections đã được tải từ trước")
            return self.collections

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
        if not self.embedding_model:
            print("Embedding model chưa được tải")
            return None

        try:
            text_embedding = self.embedding_model.encode(text, max_length=2048)["dense_vecs"]
            return text_embedding
        except Exception as e:
            print(f"Lỗi khi tạo embedding: {e}")
            return None

    def list_available_collections(self):
        """Liệt kê tất cả collections có sẵn trong Milvus"""
        # Đảm bảo kết nối được thiết lập
        if not self.connection_established:
            self.connect()

        try:
            all_collections = utility.list_collections()
            print(f"Danh sách collections có sẵn: {all_collections}")
            return all_collections
        except Exception as e:
            print(f"Lỗi khi liệt kê collections: {e}")
            return []