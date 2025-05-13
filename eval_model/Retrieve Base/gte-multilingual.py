import json
import torch
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
with open("/kaggle/input/q-and-a/question_answer.json", "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

questions = []
corpus = []
for i, item in enumerate(data):
    questions.append(item["question"])
    corpus.append(item["corpus"])

# Mã hóa corpus trước
corpus_embeddings = model.encode(corpus, return_dense=True,return_sparse=True)

# Tính toán similarity và trả về top_k kết quả cho mỗi câu hỏi
def calculate_similarities(questions, corpus_embeddings, top_k_values):
    all_results = {}
    
    # Mã hóa tất cả câu hỏi
    question_embeddings = model.encode(questions, return_dense=True,return_sparse=True)
    
    # Tính toán ma trận similarity giữa tất cả questions và corpus
    similarity_matrix = question_embeddings@corpus_embeddings.T
    
    # Lấy kết quả cho từng giá trị k
    for k in top_k_values:
        results = {}
        for i in range(len(questions)):
            # Sử dụng numpy.argsort để lấy indices của top-k giá trị
            # argsort sẽ sắp xếp theo thứ tự tăng dần, nên cần lấy từ cuối mảng
            indices = np.argsort(similarity_matrix[i])[-k:][::-1]
            results[i] = indices.tolist()  # Lưu indices của corpus có rank cao nhất
        all_results[k] = results
    
    return all_results
        

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

# Đánh giá mô hình
k_values = [3, 5, 10, 100]
all_results = calculate_similarities(questions, corpus_embeddings, k_values)
metrics = calculate_metrics(all_results, k_values)

# In kết quả
for k in k_values:
    print(f"\nTop {k} results:")
    print(f"  Accuracy@{k}: {metrics[k]['accuracy']:.4f}")
    print(f"  MRR: {metrics[k]['mrr']:.4f}")
