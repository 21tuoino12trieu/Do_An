import json
import os
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
from datasets import Dataset
import numpy as np
import torch

# Tắt wandb
os.environ["WANDB_DISABLED"] = "true"

# Đọc dữ liệu từ file JSON
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Đọc dữ liệu
data = load_data("/kaggle/input/q-and-a/question_answer.json")

# Tách câu hỏi và câu trả lời
questions = []
answers = []

for item in data:
    questions.append(item["question"])
    answers.append(item["corpus"])

# # Chuẩn bị dataset theo định dạng mới
# train_dataset = Dataset.from_dict({
#     "anchor": questions,
#     "positive": answers,
# })

# # In thông tin về dataset để xác nhận
# print(f"Tổng số cặp câu hỏi-câu trả lời: {len(train_dataset)}")
# print("Ví dụ về dữ liệu:")
# for i in range(min(3, len(train_dataset))):
#     print(f"Q: {train_dataset[i]['anchor']}")
#     print(f"A: {train_dataset[i]['positive']}")
#     print("-" * 50)

# Khởi tạo mô hình
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# Khởi tạo hàm loss
# loss = losses.MultipleNegativesRankingLoss(model)

train_examples = []
texts = []
labels = []

for i in range(len(questions)):
    texts.append([questions[i], answers[i]])
    labels.append(1)  # Gán nhãn 1 cho các cặp câu hỏi-câu trả lời tương tự

import random
random.seed(42)  # Đặt seed cho việc tái lặp

for i in range(len(questions)):
    # Chọn ngẫu nhiên một câu trả lời không tương ứng với câu hỏi hiện tại
    negative_idx = random.choice([j for j in range(len(answers)) if j != i])
    texts.append([questions[i], answers[negative_idx]])
    labels.append(0)  # 0 = cặp không tương đồng

# Chuẩn bị dataset
train_dataset = Dataset.from_dict({
    "texts": texts,
    "labels": labels
})

# Option 1: ContrastiveLoss
loss = losses.ContrastiveLoss(model=model, margin=0.5)

# Option 2: OnlineContrastiveLoss (uncomment để sử dụng)
loss = losses.OnlineContrastiveLoss(model=model, margin=0.5)

# Thiết lập các tham số huấn luyện
training_args = SentenceTransformerTrainingArguments(
    output_dir="./sentence-transformer-output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # Sử dụng mixed precision training
    report_to=None,  # Không báo cáo metrics lên wandb
    run_name="sentence-transformer-training",  # Tên khác với output_dir
)

# Khởi tạo SentenceTransformerTrainer
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
    args=training_args,
)

# Bắt đầu huấn luyện
print("Bắt đầu quá trình huấn luyện...")
trainer.train()

model = SentenceTransformer('/kaggle/working/sentence-transformer-output/checkpoint-234')
with open("/kaggle/input/q-and-a/question_answer.json","r",encoding="utf-8") as json_file:
    data = json.load(json_file)
    

questions = []
corpus = []
for i, item in enumerate(data):
    questions.append(item["question"])
    corpus.append(item["corpus"])

# Mã hóa corpus trước
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Tính toán similarity và trả về top_k kết quả cho mỗi câu hỏi
def calculate_similarities(questions, corpus_embeddings, top_k_values):
    all_results = {}
    
    # Mã hóa tất cả câu hỏi
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    
    # Tính toán ma trận similarity giữa tất cả questions và corpus
    similarity_matrix = model.similarity(question_embeddings, corpus_embeddings)
    
    # Lấy kết quả cho từng giá trị k
    for k in top_k_values:
        results = {}
        for i in range(len(questions)):
            # Lấy top-k corpus có similarity cao nhất với câu hỏi
            scores, indices = torch.topk(similarity_matrix[i], k)
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