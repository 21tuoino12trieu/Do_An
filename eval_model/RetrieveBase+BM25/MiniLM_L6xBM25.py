import json
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class BM25MiniLMRetriever:
    def __init__(self, alpha=0.5):
        """
        Khởi tạo Retriever kết hợp BM25 và MiniLM-L12-v2
        
        Args:
            alpha: Trọng số kết hợp (0: chỉ BM25, 1: chỉ MiniLM)
        """
        # Tham số BM25 cố định
        self.k1 = 1.2
        self.b = 0.75
        self.alpha = alpha
        
        # Khởi tạo mô hình MiniLM
        print("Initializing paraphrase-multilingual-MiniLM-L12-v2 model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Các biến lưu dữ liệu corpus
        self.corpus = None
        self.tokenized_corpus = None
        self.corpus_embeddings = None
        self.bm25 = None
    
    def index_corpus(self, corpus):
        """
        Đánh index corpus cho cả BM25 và MiniLM model
        
        Args:
            corpus: Danh sách các văn bản
        """
        self.corpus = corpus
        
        # Index cho BM25
        print("Tokenizing corpus for BM25...")
        self.tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        # Index cho MiniLM model
        print("Encoding corpus with MiniLM...")
        self.corpus_embeddings = self.embedding_model.encode(corpus, convert_to_tensor=True)
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
        
        # Encode queries với MiniLM model
        print("Encoding queries with MiniLM...")
        question_embeddings = self.embedding_model.encode(queries, convert_to_tensor=True)
        
        # Tính similarity từ embedding
        similarity_matrix = self.embedding_model.similarity(question_embeddings, self.corpus_embeddings)
        
        # Thực hiện truy vấn
        for i, query in enumerate(tqdm(queries, desc="Retrieving documents")):
            # Lấy điểm BM25
            tokenized_query = query.split(" ")
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
            
            # Chuẩn hóa điểm BM25 (min-max scaling)
            if np.max(bm25_scores) > np.min(bm25_scores):
                bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
            
            # Lấy điểm embedding
            embedding_scores = similarity_matrix[i].cpu().numpy()
            
            # Chuẩn hóa điểm embedding (min-max scaling)
            if np.max(embedding_scores) > np.min(embedding_scores):
                embedding_scores = (embedding_scores - np.min(embedding_scores)) / (np.max(embedding_scores) - np.min(embedding_scores))
            
            # Kết hợp điểm với trọng số alpha
            combined_scores = (1 - self.alpha) * bm25_scores + self.alpha * embedding_scores
            
            # Lấy top-k kết quả
            top_indices = np.argsort(combined_scores)[-top_k:][::-1]
            results[i] = top_indices.tolist()
        
        return results

def compute_accuracy(ground_truth_id, result_ids):
    """Tính Accuracy - kiểm tra nếu ground truth nằm trong kết quả"""
    return 1 if ground_truth_id in result_ids else 0

def compute_mrr(ground_truth_id, result_ids):
    """Tính MRR - nghịch đảo của vị trí đầu tiên mà ground truth xuất hiện"""
    for rank, id in enumerate(result_ids):
        if id == ground_truth_id:
            return 1.0 / (rank + 1)
    return 0

def evaluate_retriever(data_path, alpha_values=[0, 0.3, 0.5, 0.7, 1.0], k_values=[3, 5, 10, 100]):
    """
    Đánh giá hiệu suất của BM25 kết hợp với MiniLM
    
    Args:
        data_path: Đường dẫn đến file dữ liệu
        alpha_values: Danh sách các trọng số alpha để thử nghiệm
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
    
    # Lưu kết quả
    all_results = []
    
    # Đánh giá với các alpha khác nhau
    for alpha in alpha_values:
        print(f"\n=== Đánh giá với alpha = {alpha} ===")
        
        # Khởi tạo retriever
        retriever = BM25MiniLMRetriever(alpha=alpha)
        
        # Đánh index corpus
        retriever.index_corpus(corpus)
        
        # Truy vấn với k lớn nhất
        max_k = max(k_values)
        query_results = retriever.retrieve(questions, top_k=max_k)
        
        # Tính metrics cho từng k
        for k in k_values:
            acc_sum = 0
            mrr_sum = 0
            
            for query_id, result_ids in query_results.items():
                # Trong dữ liệu mẫu, ground truth có cùng ID với query
                ground_truth_id = query_id
                
                # Lấy k kết quả đầu tiên
                top_k_results = result_ids[:k]
                
                # Tính Accuracy@k
                acc = compute_accuracy(ground_truth_id, top_k_results)
                acc_sum += acc
                
                # Tính MRR
                mrr = compute_mrr(ground_truth_id, result_ids[:k])
                mrr_sum += mrr
            
            # Tính trung bình
            accuracy = acc_sum / len(query_results)
            mrr = mrr_sum / len(query_results)
            
            # In kết quả
            print(f"  Top {k}: Accuracy={accuracy:.4f}, MRR={mrr:.4f}")
            
            # Lưu kết quả
            all_results.append({
                "alpha": alpha,
                "k": k,
                "accuracy": accuracy,
                "mrr": mrr
            })
    
    # Hiển thị kết quả tốt nhất cho mỗi k
    print("\n=== Kết quả tốt nhất cho mỗi k ===")
    for k in k_values:
        k_results = [r for r in all_results if r["k"] == k]
        if k_results:
            best_result = max(k_results, key=lambda x: x["accuracy"])
            print(f"\nTop {k}:")
            print(f"  Alpha tốt nhất: {best_result['alpha']}")
            print(f"  Accuracy: {best_result['accuracy']:.4f}")
            print(f"  MRR: {best_result['mrr']:.4f}")


# Hàm chạy thí nghiệm chỉ với BM25 và MiniLM
def run_bm25_minilm_experiment():
    """Chạy thí nghiệm kết hợp BM25 và paraphrase-multilingual-MiniLM-L12-v2"""
    data_path = "/kaggle/input/q-and-a/question_answer.json"  # Thay đổi đường dẫn này
    
    # Các giá trị alpha khác nhau để thử nghiệm
    alpha_values = [0, 0.3, 0.5, 0.7, 1.0]
    
    # Đánh giá retriever
    evaluate_retriever(
        data_path=data_path,
        alpha_values=alpha_values,
        k_values=[3, 5, 10, 100]
    )

run_bm25_minilm_experiment()