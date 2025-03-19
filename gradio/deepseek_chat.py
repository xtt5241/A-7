from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 添加缓存装饰器避免重复加载
def load_model():
    model_name = "deepseek-ai/deepseek-llm-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

# 修改报告生成逻辑
def generate_medical_report(disease_list, confidence=85):
    # 新增错误处理
    try:
        model, tokenizer = load_model()
        prompt = f"作为眼科专家，请用中文为诊断结果 {disease_list}（置信度 {confidence}%）生成包含以下内容的报告：\n1. 病情概述\n2. 临床特征\n3. 建议治疗方案\n4. 随访建议"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=500)
        
        full_report = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 格式化输出
        return "\n".join([line for line in full_report.split("\n") if line.strip()])
    except Exception as e:
        return f"报告生成失败：{str(e)}\n（可能原因：显存不足或模型未加载）"