import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

def load_data(file_path):
    # 读取数据
    data = pd.read_csv(file_path)
    # 创建 Hugging Face 数据集
    dataset = Dataset.from_pandas(data)
    return dataset

def preprocess_data(examples, tokenizer):
    # 预处理数据以适配 T5 的输入格式
    input_texts = ["classify: " + text for text in examples['Text']]
    model_inputs = tokenizer(input_texts, max_length=512, padding="max_length", truncation=True)

    # 将标签转换为模型可识别的格式
    labels = tokenizer(["1" if label == 1 else "0" for label in examples['Result']], max_length=5, padding="max_length", truncation=True).input_ids
    model_inputs['labels'] = labels

    return model_inputs

def main():
    # 初始化分词器和模型
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # 加载和预处理数据
    dataset = load_data('llmData3.csv')
    processed_dataset = dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)

    evaldataset = load_data('llmData4.csv')
    evaldataset_dataset = evaldataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        eval_dataset=processed_dataset
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
