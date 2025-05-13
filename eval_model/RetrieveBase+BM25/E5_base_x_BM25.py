import json
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class BM25E5Retriever:
    def __init__(self, alpha=0.5):
        """
        Khởi tạo Retriever kết hợp BM25 và multilingual-e5-base
        
        Args:
            alpha: Trọng số kết hợp (0: chỉ BM25, 1: chỉ E5)
        """
        # Tham số BM25 cố định
        self.k1 = 1.2
        self.b = 0.75
        self.alpha = alpha
        
        # Khởi tạo mô hình E5
        print("Initializing intfloat/multilingual-e5-base model...")
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
        
        # Chuyển mô hình đến GPU nếu có sẵn
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.embedding_model.to(device)
        
        # Các biến lưu dữ liệu corpus
        self.corpus = None
        self.tokenized_corpus = None
        self.corpus_embeddings = None
        self.bm25 = None
    
    def index_corpus(self, corpus):
        """
        Đánh index corpus cho cả BM25 và E5 model
        
        Args:
            corpus: Danh sách các văn bản
        """
        self.corpus = corpus
        
        # Index cho BM25
        print("Tokenizing corpus for BM25...")
        self.tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        # Index cho E5 model
        print("Encoding corpus with E5...")
        # E5 sử dụng tiền tố đặc biệt cho văn bản corpus
        corpus_with_prefix = [f"passage: {doc}" for doc in corpus]
        self.corpus_embeddings = self.embedding_model.encode(corpus_with_prefix, convert_to_tensor=True)
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
        
        # Encode queries với E5 model
        print("Encoding queries with E5...")
        # E5 sử dụng tiền tố đặc biệt cho câu truy vấn
        queries_with_prefix = [f"query: {q}" for q in queries]
        question_embeddings = self.embedding_model.encode(queries_with_prefix, convert_to_tensor=True)
        
        # Tính similarity từ embedding
        # Chuyển về numpy để xử lý
        if isinstance(question_embeddings, torch.Tensor):
            question_embeddings_np = question_embeddings.cpu().numpy()
        else:
            question_embeddings_np = question_embeddings
            
        if isinstance(self.corpus_embeddings, torch.Tensor):
            corpus_embeddings_np = self.corpus_embeddings.cpu().numpy()
        else:
            corpus_embeddings_np = self.corpus_embeddings
        
        # Tính dotproduct similarity matrix
        similarity_matrix = np.matmul(question_embeddings_np, corpus_embeddings_np.T)
        
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

# Tính ACC và MRR
def compute_accuracy(ground_truth_id, result_ids):
    # Một câu hỏi đúng khi câu trả lời đúng (cùng index) nằm trong top kết quả
    return 1 if ground_truth_id in result_ids else 0

def compute_mrr(ground_truth_id, result_ids):
    # Tính MRR dựa trên vị trí của câu trả lời đúng trong kết quả
    for rank, id in enumerate(result_ids):
        if id == ground_truth_id:
            return 1.0 / (rank + 1)
    return 0

def calculate_metrics(all_results, k_values):
    metrics = {}
    
    for k in k_values:
        results = all_results[k]
        sum_acc = 0
        sum_mrr = 0
        
        for query_id, result_ids in results.items():
            # Trong dataset này, câu trả lời đúng cho câu hỏi có index i là corpus có index i
            ground_truth_id = query_id
            
            acc = compute_accuracy(ground_truth_id, result_ids)
            mrr = compute_mrr(ground_truth_id, result_ids)
            
            sum_acc += acc
            sum_mrr += mrr
        
        # Tính giá trị trung bình
        metrics[k] = {
            "accuracy": sum_acc / len(results),
            "mrr": sum_mrr / len(results)
        }
    
    return metrics

# Hàm chính để đánh giá kết hợp BM25 và E5
def evaluate_bm25_e5(data_path, alpha_values=[0, 0.3, 0.5, 0.7, 1.0], k_values=[3, 5, 10, 100]):
    """
    Đánh giá hiệu suất của BM25 kết hợp với multilingual-e5-base
    
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
    
    # Đánh giá với các alpha khác nhau
    for alpha in alpha_values:
        print(f"\n=== Đánh giá với alpha = {alpha} ===")
        
        # Khởi tạo retriever
        retriever = BM25E5Retriever(alpha=alpha)
        
        # Đánh index corpus
        retriever.index_corpus(corpus)
        
        # Truy vấn với k lớn nhất
        max_k = max(k_values)
        query_results = retriever.retrieve(questions, top_k=max_k)
        
        # Chuyển đổi kết quả cho tương thích với calculate_metrics
        all_results = {}
        for k in k_values:
            all_results[k] = {}
            for query_id, result_ids in query_results.items():
                all_results[k][query_id] = result_ids[:k]
        
        # Tính metrics
        metrics = calculate_metrics(all_results, k_values)
        
        # In kết quả
        for k in k_values:
            print(f"\nTop {k} results (alpha={alpha}):")
            print(f"  Accuracy@{k}: {metrics[k]['accuracy']:.4f}")
            print(f"  MRR: {metrics[k]['mrr']:.4f}")

# Thực thi đánh giá
if __name__ == "__main__":
    data_path = "/kaggle/input/q-and-a/question_answer.json"  # Thay đổi đường dẫn này
    
    # Các giá trị alpha khác nhau để thử nghiệm
    alpha_values = [0, 0.3, 0.5, 0.7, 1.0]
    
    # Đánh giá kết hợp BM25 và E5
    evaluate_bm25_e5(
        data_path=data_path,
        alpha_values=alpha_values,
        k_values=[3, 5, 10, 100]
    )