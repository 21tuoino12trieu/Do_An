"""
File kiểm tra đơn giản cho việc embedding query và tìm kiếm ngữ nghĩa
trên trường content trong Milvus
"""

import numpy as np
from pymilvus import connections, Collection
from FlagEmbedding import BGEM3FlagModel
import time

# Cấu hình kết nối
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "product_database_content"
EMBEDDING_MODEL = "BAAI/bge-m3"


def connect_to_milvus():
    """Kết nối đến Milvus server"""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
        )
        print("✅ Kết nối thành công đến Milvus")
        return True
    except Exception as e:
        print(f"❌ Lỗi kết nối đến Milvus: {e}")
        return False


def load_embedding_model():
    """Tải model embedding"""
    try:
        start_time = time.time()
        model = BGEM3FlagModel(EMBEDDING_MODEL, use_cuda=False)
        print(f"✅ Đã tải model embedding ({time.time() - start_time:.2f}s)")
        return model
    except Exception as e:
        print(f"❌ Lỗi khi tải model embedding: {e}")
        return None


def load_collection():
    """Tải collection content"""
    try:
        collection = Collection(name=COLLECTION_NAME)
        collection.load()
        print(f"✅ Đã tải collection {COLLECTION_NAME}")

        # In thông tin schema để kiểm tra chiều của vector
        schema = collection.schema
        for field in schema.fields:
            if field.name == "embedding":
                print(
                    f"📏 Chiều vector trong collection: {field.params['dim'] if 'dim' in field.params else 'không xác định'}")

        return collection
    except Exception as e:
        print(f"❌ Lỗi khi tải collection: {e}")
        return None


def get_embedding(text, model):
    """Tạo embedding từ text và in thông tin chi tiết về kích thước"""
    try:
        start_time = time.time()
        embedding_result = model.encode(text, max_length=2048)
        dense_vecs = embedding_result["dense_vecs"]

        # In thông tin chi tiết về embedding
        print(f"📊 Loại dữ liệu embedding: {type(dense_vecs)}")

        if isinstance(dense_vecs, np.ndarray):
            print(f"📊 Kích thước embedding: {dense_vecs.shape}")
            print(f"📊 Kiểu dữ liệu: {dense_vecs.dtype}")
        else:
            print(f"📊 Độ dài embedding: {len(dense_vecs)}")

        print(f"✅ Đã tạo embedding ({time.time() - start_time:.2f}s)")
        return dense_vecs
    except Exception as e:
        print(f"❌ Lỗi khi tạo embedding: {e}")
        return None


def semantic_search(query, collection, model, top_k=5):
    """Thực hiện tìm kiếm ngữ nghĩa với nhiều cách chuyển đổi khác nhau"""
    try:
        print("\n🔍 Bắt đầu tìm kiếm cho query:", query)

        # Lấy embedding gốc
        raw_embedding = get_embedding(query, model)
        if raw_embedding is None:
            return

        # Danh sách các biến thể chuyển đổi để thử
        embedding_variants = []

        # Biến thể 1: Nguyên bản
        embedding_variants.append(("Nguyên bản", raw_embedding))

        # Biến thể 2: Chuyển đổi sang np.float32
        if isinstance(raw_embedding, np.ndarray) and raw_embedding.dtype != np.float32:
            embedding_variants.append(("Float32", raw_embedding.astype(np.float32)))

        # Biến thể 3: Chuyển đổi sang list
        if isinstance(raw_embedding, np.ndarray):
            embedding_variants.append(("List", raw_embedding.tolist()))

        # Biến thể 4: Đảm bảo 1D vector nếu cần
        if isinstance(raw_embedding, np.ndarray) and len(raw_embedding.shape) > 1:
            if raw_embedding.shape[0] == 1:  # Nếu là ma trận 2D với 1 hàng
                embedding_variants.append(("1D Vector", raw_embedding[0]))

        # Thử từng biến thể
        for variant_name, query_embedding in embedding_variants:
            print(f"\n🧪 Thử với biến thể: {variant_name}")
            print(f"📊 Loại: {type(query_embedding)}")

            if isinstance(query_embedding, np.ndarray):
                print(f"📊 Kích thước: {query_embedding.shape}")
                print(f"📊 Kiểu dữ liệu: {query_embedding.dtype}")
            else:
                print(f"📊 Độ dài: {len(query_embedding)}")

            try:
                search_params = {"metric_type": "IP", "params": {"ef": 256}}

                # Đảm bảo rằng query_embedding là list khi truyền vào search
                search_data = [query_embedding]

                start_time = time.time()
                results = collection.search(
                    data=search_data,
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    output_fields=["product_name", "chunk_id", "text_data"]
                )

                print(f"✅ Tìm kiếm thành công ({time.time() - start_time:.2f}s)")
                print(f"🔢 Số kết quả: {len(results[0])}")
                print(results[0])
                # In 2 kết quả đầu tiên
                for i, hit in enumerate(results[0][:2]):
                    print(f"\n📱 Kết quả #{i + 1}: {hit.entity.get('product_name')}")
                    print(f"🔍 Điểm số: {hit.score:.4f}")

                # Biến thể này thành công, không cần thử các biến thể khác
                return [{
                    "product_name": hit.entity.get("product_name"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "text_data": hit.entity.get("text_data"),
                    "score": hit.score
                } for hit in results[0]]

            except Exception as e:
                print(f"❌ Lỗi khi thử biến thể {variant_name}: {e}")
                continue

        print("\n❌ Tất cả các biến thể đều thất bại")
        return None

    except Exception as e:
        print(f"❌ Lỗi trong quá trình tìm kiếm: {e}")
        return None


def main():
    """Hàm chính để thực hiện test"""
    # Kết nối và tải dữ liệu
    if not connect_to_milvus():
        return

    collection = load_collection()
    if collection is None:
        return

    model = load_embedding_model()
    if model is None:
        return

    # Các câu query để test
    test_queries = [
        "Tai nghe chống ồn tốt nhất",
    ]

    # Thực hiện tìm kiếm cho từng câu query
    for query in test_queries:
        results = semantic_search(query, collection, model)
        if results:
            print(f"\n✅ Tìm kiếm thành công cho query: '{query}'")
        else:
            print(f"\n❌ Tìm kiếm thất bại cho query: '{query}'")

    # Đóng kết nối
    try:
        connections.disconnect("default")
        print("\n✅ Đã đóng kết nối Milvus")
    except:
        print("\n❌ Lỗi khi đóng kết nối Milvus")


if __name__ == "__main__":
    main()