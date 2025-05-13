import json
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

class BM25EmbeddingRetriever:
    def __init__(self, embedding_model_name='BAAI/bge-m3', alpha=0.5):
        """
        Khởi tạo Retriever kết hợp BM25 và embedding model
        
        Args:
            embedding_model_name: Tên của mô hình embedding
            alpha: Trọng số kết hợp (0: chỉ BM25, 1: chỉ embedding)
        """
        # Tham số BM25 cố định
        self.k1 = 1.2
        self.b = 0.75
        self.alpha = alpha
        
        # Khởi tạo mô hình embedding
        self.embedding_model = BGEM3FlagModel(embedding_model_name, use_fp16=True)
        
        # Các biến lưu dữ liệu corpus
        self.corpus = None
        self.tokenized_corpus = None
        self.corpus_embeddings = None
        self.bm25 = None
    
    def index_corpus(self, corpus):
        """
        Đánh index corpus cho cả BM25 và embedding model
        
        Args:
            corpus: Danh sách các văn bản
        """
        self.corpus = corpus
        
        # Index cho BM25
        print("Tokenizing corpus for BM25...")
        self.tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        # Index cho embedding model
        print("Encoding corpus with embedding model...")
        self.corpus_embeddings = self.embedding_model.encode(corpus)['dense_vecs']
        print(f"Corpus embeddings shape: {self.corpus_embeddings.shape}")
    
    def retrieve(self, queries, top_k=10):
        """
        Truy xuất văn bản liên quan cho các truy vấn
        
        Args:
            queries: Danh sách các câu truy vấn
            top_k: Số lượng kết quả trả về
            
        Returns:
            Dict chứa kết quả cho mỗi query
        """
        if self.corpus is None:
            raise ValueError("Corpus chưa được đánh index. Hãy gọi index_corpus() trước.")
        
        results = {}
        
        # Encode queries với mô hình embedding
        print("Encoding queries with embedding model...")
        query_embeddings = self.embedding_model.encode(queries)['dense_vecs']
        
        # Tính similarity từ embedding
        similarity_matrix = query_embeddings @ self.corpus_embeddings.T
        
        # Thực hiện truy vấn
        for i, query in enumerate(tqdm(queries, desc="Retrieving documents")):
            # Lấy điểm BM25
            tokenized_query = query.split(" ")
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
            
            # Chuẩn hóa điểm BM25 (min-max scaling)
            if np.max(bm25_scores) > np.min(bm25_scores):
                bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
            
            # Lấy điểm embedding
            embedding_scores = similarity_matrix[i]
            
            # Chuẩn hóa điểm embedding (min-max scaling)
            if np.max(embedding_scores) > np.min(embedding_scores):
                embedding_scores = (embedding_scores - np.min(embedding_scores)) / (np.max(embedding_scores) - np.min(embedding_scores))
            
            # Kết hợp điểm với trọng số alpha
            combined_scores = (1 - self.alpha) * bm25_scores + self.alpha * embedding_scores
            
            # Lấy top-k kết quả
            top_indices = np.argsort(combined_scores)[-top_k:][::-1]
            results[i] = top_indices.tolist()
        
        return results

# Hàm chính để chạy thí nghiệm
def evaluate_combined_model(data_path, alpha=0.5, k_values=[3, 5, 10, 100]):
    """
    Đánh giá hiệu suất của mô hình kết hợp và so sánh với BM25 và Embedding riêng lẻ
    
    Args:
        data_path: Đường dẫn đến file dữ liệu
        alpha: Trọng số kết hợp (0: chỉ BM25, 1: chỉ embedding)
        k_values: Danh sách các giá trị k để đánh giá
    """
    # Đọc dữ liệu
    with open(data_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    # Chuẩn bị dữ liệu
    questions = []
    corpus = []
    for i, item in enumerate(data):
        questions.append(item["question"])
        corpus.append(item["corpus"])
    
    # Khởi tạo retriever kết hợp
    combined_retriever = BM25EmbeddingRetriever(embedding_model_name='BAAI/bge-m3', alpha=alpha)
    combined_retriever.index_corpus(corpus)
    
    # Đánh giá mô hình kết hợp
    combined_results = evaluate_model(combined_retriever, questions, corpus, k_values)
    
    # Đánh giá riêng BM25
    bm25_retriever = BM25EmbeddingRetriever(embedding_model_name='BAAI/bge-m3', alpha=0.0)
    bm25_retriever.index_corpus(corpus)
    bm25_results = evaluate_model(bm25_retriever, questions, corpus, k_values)
    
    # Đánh giá riêng Embedding
    embedding_retriever = BM25EmbeddingRetriever(embedding_model_name='BAAI/bge-m3', alpha=1.0)
    embedding_retriever.index_corpus(corpus)
    embedding_results = evaluate_model(embedding_retriever, questions, corpus, k_values)
    
    # In kết quả so sánh
    print("\n=== Kết quả so sánh các phương pháp ===")
    
    for k in k_values:
        print(f"\nTop-{k} results:")
        print(f"  BM25 only (k1=1.2, b=0.75): Accuracy={bm25_results[k]['accuracy']:.4f}, MRR={bm25_results[k]['mrr']:.4f}")
        print(f"  Embedding only: Accuracy={embedding_results[k]['accuracy']:.4f}, MRR={embedding_results[k]['mrr']:.4f}")
        print(f"  Combined (alpha={alpha}): Accuracy={combined_results[k]['accuracy']:.4f}, MRR={combined_results[k]['mrr']:.4f}")

def evaluate_model(retriever, queries, corpus, k_values):
    """
    Đánh giá hiệu suất của một retriever
    
    Args:
        retriever: Đối tượng retriever
        queries: Danh sách câu truy vấn
        corpus: Danh sách văn bản
        k_values: Danh sách các giá trị k để đánh giá
    
    Returns:
        Dict chứa các metrics theo k
    """
    # Lấy kết quả truy vấn
    max_k = max(k_values)
    query_results = retriever.retrieve(queries, top_k=max_k)
    
    # Tính metrics cho từng k
    metrics = {}
    for k in k_values:
        acc_sum = 0
        mrr_sum = 0
        
        for query_id, result_ids in query_results.items():
            # Trong dữ liệu mẫu, ground truth có cùng ID với query
            ground_truth_id = query_id
            
            # Lấy k kết quả đầu tiên
            top_k_results = result_ids[:k]
            
            # Tính Accuracy@k
            acc = 1 if ground_truth_id in top_k_results else 0
            acc_sum += acc
            
            # Tính MRR
            mrr = 0
            for rank, doc_id in enumerate(result_ids):
                if doc_id == ground_truth_id:
                    mrr = 1.0 / (rank + 1)
                    break
            mrr_sum += mrr
        
        # Tính trung bình
        metrics[k] = {
            "accuracy": acc_sum / len(query_results),
            "mrr": mrr_sum / len(query_results)
        }
    
    return metrics

# Hàm chạy thí nghiệm với nhiều giá trị alpha khác nhau
def run_alpha_experiment(data_path, k=10):
    """
    Chạy thí nghiệm với nhiều giá trị alpha khác nhau để tìm kết quả tốt nhất
    
    Args:
        data_path: Đường dẫn đến file dữ liệu
        k: Giá trị k để đánh giá
    """
    # Đọc dữ liệu
    with open(data_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    # Chuẩn bị dữ liệu
    questions = []
    corpus = []
    for i, item in enumerate(data):
        questions.append(item["question"])
        corpus.append(item["corpus"])
    
    # Khởi tạo retriever cơ bản
    base_retriever = BM25EmbeddingRetriever(embedding_model_name='BAAI/bge-m3')
    base_retriever.index_corpus(corpus)
    
    # Thử nghiệm với các giá trị alpha khác nhau
    alpha_values = [0, 0.3, 0.5, 0.7, 1.0]
    results = []
    
    for alpha in alpha_values:
        base_retriever.alpha = alpha
        query_results = base_retriever.retrieve(questions, top_k=k)
        
        # Tính metrics
        acc_sum = 0
        mrr_sum = 0
        
        for query_id, result_ids in query_results.items():
            ground_truth_id = query_id
            top_k_results = result_ids[:k]
            
            acc = 1 if ground_truth_id in top_k_results else 0
            acc_sum += acc
            
            mrr = 0
            for rank, doc_id in enumerate(result_ids):
                if doc_id == ground_truth_id:
                    mrr = 1.0 / (rank + 1)
                    break
            mrr_sum += mrr
        
        # Lưu kết quả
        results.append({
            "alpha": alpha,
            "accuracy": acc_sum / len(query_results),
            "mrr": mrr_sum / len(query_results)
        })
    
    # In kết quả
    print(f"\n=== Kết quả thí nghiệm với các giá trị alpha (Top-{k}) ===")
    for result in results:
        print(f"  Alpha={result['alpha']:.1f}: Accuracy={result['accuracy']:.4f}, MRR={result['mrr']:.4f}")
    
    # Tìm alpha tốt nhất
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\nGiá trị alpha tốt nhất: {best_result['alpha']:.1f} với Accuracy={best_result['accuracy']:.4f}, MRR={best_result['mrr']:.4f}")

data_path = "/kaggle/input/q-and-a/question_answer.json"
 
# Tìm giá trị alpha tốt nhất
k_values = [3,5,10,100]
for item in k_values:
    run_alpha_experiment(data_path, k=item)