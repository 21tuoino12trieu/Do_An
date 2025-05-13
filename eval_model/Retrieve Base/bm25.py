import json
import numpy as np
from rank_bm25 import BM25Okapi

with open("/kaggle/input/q-and-a/question_answer.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = []
corpus = []
for i, item in enumerate(data):
    questions.append(item["question"])
    corpus.append(item["corpus"])

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

# Tokenize corpus for BM25
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def find_similar_queries_bm25(queries, bm25_model, top_k_values):
    all_results = {}
    
    for k in top_k_values:
        results = {}
        for index, query in enumerate(queries):
            tokenized_query = query.split(" ")
            doc_scores = bm25_model.get_scores(tokenized_query)
            top_k_indices = np.argsort(doc_scores)[-k:][::-1]
            results[index] = top_k_indices.tolist()
        all_results[k] = results
    
    return all_results

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
all_results = find_similar_queries_bm25(questions, bm25, k_values)
metrics = calculate_metrics(all_results, k_values)

# In kết quả
for k in k_values:
    print(f"\nTop {k} results:")
    print(f"  Accuracy@{k}: {metrics[k]['accuracy']:.4f}")
    print(f"  MRR: {metrics[k]['mrr']:.4f}")

# In chi tiết một số kết quả
print("\nDetailed results for first 5 questions (or all if less than 5):")
for i in range(min(5, len(questions))):
    print(f"Question {i+1}: '{questions[i][:50]}...'")
    for k in k_values:
        result_ids = all_results[k][i]
        ground_truth_id = i  # Câu trả lời đúng có index i
        in_top_k = ground_truth_id in result_ids
        rank = result_ids.index(ground_truth_id) + 1 if in_top_k else f"Not in top {k}"
        print(f"  In top {k}: {'Yes' if in_top_k else 'No'} (Rank: {rank})")