from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def is_phishing_email(email_text):
    # 加载预训练的GPT模型和分词器
    model_name = "EleutherAI/gpt-neo-125M"
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 构造prompt
    prompt = f"Email: \"{email_text}\" Is this a phishing email? Answer yes or no."
    
    # 编码并生成文本
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

    # 解码生成的文本
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 简单的方式来判断输出
    return "yes" in decoded_output.lower()

# 测试邮件文
test_email = "You have won $10,000! Click here to claim your prize now!"
test_email2 = "Dear user, your account will be suspended unless you re-enter your credentials by following this link."
result = is_phishing_email(test_email)

print(f"Is the email a phishing attempt? {result}")
