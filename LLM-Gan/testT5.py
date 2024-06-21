from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
model_path = r'F:\592\results\checkpoint-2'
model = T5ForConditionalGeneration.from_pretrained(model_path)  # 假设模型保存在这个路径
tokenizer = T5Tokenizer.from_pretrained(model_path)



def load_data(file_path):
    # 读取数据
    data = pd.read_csv(file_path)
    # 创建 Hugging Face 数据集
    dataset = Dataset.from_pandas(data)
    return dataset

test_dataset = load_data('llmData4.csv')
test_dataloader = DataLoader(test_dataset, batch_size=16)  # `test_dataset` 是你的测试数据集
import torch

model.eval()  # 将模型设置为评估模式
correct = 0
total = 0

with torch.no_grad():
    for batch in test_dataloader:
        inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(inputs['input_ids'])
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # 比较 predictions 和 batch['labels'] 来计算正确率
        correct += (predictions == batch['labels']).sum().item()
        total += len(batch['labels'])

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")
