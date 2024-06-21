from transformers import BertForSequenceClassification, BertTokenizer
import torch

def predict_phishing(email_text):
    # 加载预训练的BERT模型和分词器
    model_name = 'bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 预处理邮件文本
    inputs = tokenizer(email_text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)

    # 输出预测结果，1为钓鱼邮件，0为非钓鱼邮件
    return 'Phishing' if prediction.item() == 1 else 'Not Phishing'

# 测试邮件文本
test_email = "You have won $10,000! Click here to claim your prize now!"
test_email2 = "Dear user, your account will be suspended unless you re-enter your credentials by following this link."
result = predict_phishing(test_email2)

print(f"Email classification: {result}")
