from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1️⃣ 加载 DeepSeek-Chat 本地模型
model_name = "deepseek-ai/deepseek-llm-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 2️⃣ 生成医学报告
def generate_medical_report(disease, confidence):
    prompt = f"病人眼底检查结果：AI 诊断为 {disease}，置信度 {confidence}%。请给出详细的病情描述和医生建议。"
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_length=300)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 测试代码
if __name__ == "__main__":
    print(generate_medical_report("糖尿病视网膜病变", 92))
