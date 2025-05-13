import json
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

class BM25MiniLMRetriever:
    def __init__(self, alpha=0.7,
                use_reranker=False, 
                reranker_model_name="BAAI/bge-reranker-v2-m3"):

        self.alpha = alpha
        self.k1 = 1.2
        self.b = 0.75
        # Khởi tạo mô hình MiniLM
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Chuyển mô hình vào GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.embedding_model.to(device)
        
        self.use_reranker = use_reranker
        if use_reranker:
            print(f"Initializing reranker {reranker_model_name}")
            self.reranker = FlagReranker(reranker_model_name, use_fp16=True)
        else:
            self.reranker = None

    def index_corpus(self, corpus):
        self.corpus = corpus
        self.tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        print("Encoding corpus with MiniLM...")
        self.corpus_embeddings = self.embedding_model.encode(corpus, convert_to_tensor=True)
        print(f"Corpus embeddings shape: {self.corpus_embeddings.shape}")

    def retrieve(self, queries, top_k=10, rerank_top_k=100):
        if self.corpus is None:
            raise ValueError("Corpus chưa được đánh index. Gọi index_corpus() trước.")
        results = {}
        print("Encoding queries with MiniLM...")
        question_embeddings = self.embedding_model.encode(queries, convert_to_tensor=True)

        if isinstance(question_embeddings, torch.Tensor):
            question_embeddings_np = question_embeddings.cpu().numpy()
        else:
            question_embeddings_np = question_embeddings

        if isinstance(self.corpus_embeddings, torch.Tensor):
            corpus_embeddings_np = self.corpus_embeddings.cpu().numpy()
        else:
            corpus_embeddings_np = self.corpus_embeddings
        
        similarity_matrix = np.matmul(question_embeddings_np, corpus_embeddings_np.T)

        for i, query in enumerate(queries):
            tokenized_query = query.split(" ")
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

            if np.max(bm25_scores) > np.min(bm25_scores):
                bm25_scores = (bm25_scores - np.min(bm25_scores))/(np.max(bm25_scores) - np.min(bm25_scores))

            embedding_scores = similarity_matrix[i] 

            if np.max(embedding_scores) > np.min(embedding_scores):
                embedding_scores = (embedding_scores - np.min(embedding_scores))/(np.max(embedding_scores) - np.min(embedding_scores))   
            
            combined_scores = (1-self.alpha) * bm25_scores + self.alpha * embedding_scores
            k_to_retrieve = rerank_top_k if self.use_reranker else top_k
            top_indices = np.argsort(combined_scores)[-k_to_retrieve:][::-1]
            
            if self.use_reranker:
                candidate_docs = [self.corpus[idx] for idx in top_indices]
                rerank_scores = self.reranker.compute_score([[query, doc] for doc in candidate_docs])
                reranked_indices = [top_indices[i] for i in np.argsort(rerank_scores)[::-1][:top_k]]
                results[i] = reranked_indices
            else:
                results[i] = top_indices[:top_k].tolist()
        
        return results

def compute_accuracy(ground_truth_id, result_ids):
    return 1 if ground_truth_id in result_ids else 0
    
def compute_mrr(ground_truth_id, result_ids):
    for rank, id in enumerate(result_ids):
        if id == ground_truth_id:
            return 1.0/(rank+1)
    return 0.0
    
def calculate_metrics(all_results, k_values):
    metrics = {}
    for k in k_values:
        results = all_results[k]
        sum_acc = 0
        sum_mrr = 0
        for query_id, result_ids in results.items():
            ground_truth_id = query_id
            acc = compute_accuracy(ground_truth_id, result_ids)
            mrr = compute_mrr(ground_truth_id, result_ids)
            sum_acc += acc
            sum_mrr += mrr
        metrics[k] = {
            "accuracy": sum_acc / len(results),
            "mrr": sum_mrr / len(results)
        }
    return metrics

def evaluate_bm25_minilm_with_reranker(data_path, alpha_values=[0, 0.3, 0.5, 0.7, 1.0], k_values=[1, 3, 5, 10, 100], use_reranker=False):
    with open(data_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    questions = []
    corpus = []
    for i, item in enumerate(data):
        questions.append(item["question"])
        corpus.append(item["corpus"])
    
    reranker_status = "với Reranker" if use_reranker else "không Reranker"
    print(f"\n=== Kết quả đánh giá BM25 và MiniLM {reranker_status} ===")
    
    for alpha in alpha_values:
        print(f"\n=== Đánh giá với alpha = {alpha} {reranker_status} ===")

        retriever = BM25MiniLMRetriever(alpha=alpha, use_reranker=use_reranker)
        retriever.index_corpus(corpus)
        
        max_k = max(k_values)
        query_results = retriever.retrieve(questions, top_k=max_k, rerank_top_k=100)
        
        all_results = {}
        for k in k_values:
            all_results[k] = {}
            for query_id, result_ids in query_results.items():
                all_results[k][query_id] = result_ids[:k]
        
        metrics = calculate_metrics(all_results, k_values)
        
        for k in k_values:
            print(f"\nTop {k} results (alpha={alpha}):")
            print(f"  Accuracy@{k}: {metrics[k]['accuracy']:.4f}")
            print(f"  MRR: {metrics[k]['mrr']:.4f}")

def compare_with_without_reranker(data_path, alpha_values=[0, 0.3, 0.5, 0.7, 1.0], k_values=[1, 3, 5, 10, 100]):
    """
    So sánh hiệu suất của mô hình với và không có reranker
    """
    print("\n=== So sánh hiệu suất với và không có Reranker ===")
    
    # Đánh giá không có reranker
    evaluate_bm25_minilm_with_reranker(
        data_path=data_path,
        alpha_values=alpha_values,
        k_values=k_values,
        use_reranker=False
    )
    
    # Đánh giá có reranker
    evaluate_bm25_minilm_with_reranker(
        data_path=data_path,
        alpha_values=alpha_values,
        k_values=k_values,
        use_reranker=True
    )

# Thực thi đánh giá
if __name__ == "__main__":
    data_path = "/kaggle/input/q-and-a/question_answer.json"  # Thay đổi đường dẫn này nếu cần
    
    # Các giá trị alpha khác nhau để thử nghiệm
    alpha_values = [0, 0.3, 0.5, 0.7, 1.0]
    k_values = [1, 3, 5, 10, 100]
    
    # So sánh kết quả với và không có reranker
    compare_with_without_reranker(
        data_path=data_path,
        alpha_values=alpha_values,
        k_values=k_values
    )