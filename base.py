import json
from FlagEmbedding import BGEM3FlagModel
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import hashlib

# Cách tạo vector database bằng Milvus:
# + File ban đầu có rất nhiều sản phẩm
# + Mỗi sản phẩm có đến 7 khía cạnh riêng
# + 7 khía cạnh riêng tương đương với 7 fields tương đương với việc tạo ra 7 collections
# + Mỗi collection gồm 5 cột: id, product_name, chunk_id, text_data, embedding
# -> embedding được đánh index bằng HNSW và Inner Product

def connect_to_milvus(host="localhost", port="19530"):
    connections.connect(
        alias="default",
        host=host,
        port=port,
    )
    print("Đã kết nối được milvus")

def compute_arg_hash(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**63)

def create_collection_for_field(base_name, field_name, dim=1024):
    collection_name = f"{base_name}_{field_name}"

    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} đã tồn tại.")
        return Collection(name=collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_increment=False),
        FieldSchema(name="product_name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="text_data", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields=fields, description=f"Collection for {field_name} embeddings")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 32, "efConstruction": 256}
    }

    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Đã tạo index cho collection {collection_name}")

    return collection


def create_all_collections(base_name="product_database", dim=1024):
    """Tạo tất cả các collection cần thiết"""
    collections = {}
    for field in ["product_name", "product_info", "warranty", "technical", "feature", "content", "full_promotion"]:
        collections[field] = create_collection_for_field(base_name, field, dim)

    return collections


def chunk_text(text, chunk_size=1200, over_lap=100):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - over_lap):
        chunk = tokens[i: i + chunk_size]
        chunks.append(" ".join(chunk))

        if i + chunk_size >= len(tokens):
            break
    return chunks


def load_bge_model(model_name="BAAI/bge-m3"):
    model = BGEM3FlagModel(model_name, use_cuda=True)
    return model


def get_embeddings(text, model):
    text_embedding = model.encode(text, max_length=2048)["dense_vecs"]
    return text_embedding


def process_json_data(json_file_path, model, chunk_content=True, chunk_size=1200, over_lap=100):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = {
        "product_name": [],
        "product_info": [],
        "warranty": [],
        "technical": [],
        "feature": [],
        "content": [],
        "full_promotion": []
    }

    total_items = len(data)

    for i, item in enumerate(data):
        print(f"Đang xử lí item {i + 1}/{total_items}")

        product_name = item["product_name"]
        product_info = item["product_info_summary"]
        warranty = item["warranty_summary"]
        technical = item["technical_summary"]
        feature = item["feature_summary"]
        content = item["content_summary"]
        full_promotion = item["full_promotion"]

        # Embed các trường không thay đổi theo chunk
        product_name_emb = get_embeddings(product_name, model)
        product_info_emb = get_embeddings(product_info, model)
        warranty_emb = get_embeddings(warranty, model)
        technical_emb = get_embeddings(technical, model)
        feature_emb = get_embeddings(feature, model)
        full_promotion_emb = get_embeddings(full_promotion, model)

        # Thêm vào danh sách tương ứng
        processed_data["product_name"].append({
            "product_name": product_name,
            "chunk_id": 0,
            "text_data": product_name,
            "embedding": product_name_emb
        })

        processed_data["product_info"].append({
            "product_name": product_name,
            "chunk_id": 0,
            "text_data": product_info,
            "embedding": product_info_emb
        })

        processed_data["warranty"].append({
            "product_name": product_name,
            "chunk_id": 0,
            "text_data": warranty,
            "embedding": warranty_emb
        })

        processed_data["technical"].append({
            "product_name": product_name,
            "chunk_id": 0,
            "text_data": technical,
            "embedding": technical_emb
        })

        processed_data["feature"].append({
            "product_name": product_name,
            "chunk_id": 0,
            "text_data": feature,
            "embedding": feature_emb
        })

        processed_data["full_promotion"].append({
            "product_name": product_name,
            "chunk_id": 0,
            "text_data": full_promotion,
            "embedding": full_promotion_emb
        })

        # Chunk content nếu dài
        if chunk_content and len(content.split()) > chunk_size:
            content_chunks = chunk_text(content, chunk_size, over_lap)
        else:
            content_chunks = [content]

        for idx, content_chunk in enumerate(content_chunks):
            content_emb = get_embeddings(content_chunk, model)

            processed_data["content"].append({
                "product_name": product_name,
                "chunk_id": idx,
                "text_data": content_chunk,
                "embedding": content_emb
            })

    return processed_data


def insert_to_collections(collections, processed_data):
    for field, items in processed_data.items():
        collection = collections[field]

        ids = []
        product_names = []
        chunk_ids = []
        text_data_list = []
        embeddings = []

        for i, item in enumerate(items):
            product_names.append(item["product_name"])
            ids.append(compute_arg_hash(product_names[i]))
            chunk_ids.append(item["chunk_id"])
            text_data_list.append(item["text_data"])
            embeddings.append(item["embedding"])

        if product_names:
            collection.insert([
                ids,
                product_names,
                chunk_ids,
                text_data_list,
                embeddings
            ])
            collection.flush()
            print(f"✅ Đã insert dữ liệu vào collection {collection.name}")


def process_test_embed_data(json_file_path):
    """Xử lý dữ liệu đã có sẵn embedding"""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = {
        "product_name": [],
        "product_info": [],
        "warranty": [],
        "technical": [],
        "feature": [],
        "content": [],
        "full_promotion": []
    }

    for item in data:
        # Thêm vào danh sách tương ứng với cấu trúc mới
        processed_data["product_name"].append({
            "product_name": item["product_name"],
            "chunk_id": item["chunk_id"],
            "text_data": item["product_name"],
            "embedding": np.array(item["product_name_embedding"], dtype=np.float32)
        })

        processed_data["product_info"].append({
            "product_name": item["product_name"],
            "chunk_id": item["chunk_id"],
            "text_data": item["product_info_summary"],
            "embedding": np.array(item["product_info_embedding"], dtype=np.float32)
        })

        processed_data["warranty"].append({
            "product_name": item["product_name"],
            "chunk_id": item["chunk_id"],
            "text_data": item["warranty_summary"],
            "embedding": np.array(item["warranty_embedding"], dtype=np.float32)
        })

        processed_data["technical"].append({
            "product_name": item["product_name"],
            "chunk_id": item["chunk_id"],
            "text_data": item["technical_summary"],
            "embedding": np.array(item["technical_embedding"], dtype=np.float32)
        })

        processed_data["feature"].append({
            "product_name": item["product_name"],
            "chunk_id": item["chunk_id"],
            "text_data": item["feature_summary"],
            "embedding": np.array(item["feature_embedding"], dtype=np.float32)
        })

        processed_data["content"].append({
            "product_name": item["product_name"],
            "chunk_id": item["chunk_id"],
            "text_data": item["content_summary"],
            "embedding": np.array(item["content_embedding"], dtype=np.float32)
        })

        processed_data["full_promotion"].append({
            "product_name": item["product_name"],
            "chunk_id": item["chunk_id"],
            "text_data": item["full_promotion"],
            "embedding": np.array(item["full_promotion_embedding"], dtype=np.float32)
        })

    return processed_data


def main(base_name="product_database"):
    try:
        # Kết nối Milvus
        connect_to_milvus()

        # Tạo các collection riêng biệt cho từng loại embedding
        collections = create_all_collections(base_name)

        # Đường dẫn file JSON của bạn
        json_file_path = "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\merged_data.json"

        # Xử lý dữ liệu đã có sẵn embedding
        processed_data = process_test_embed_data(json_file_path)

        # Insert dữ liệu vào các collection
        insert_to_collections(collections, processed_data)

        # Load các collection để sẵn sàng tìm kiếm
        for collection in collections.values():
            collection.load()

        print(f"Các collection đã sẵn sàng cho việc tìm kiếm")

        return collections
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

if __name__ == "__main__":
    main()