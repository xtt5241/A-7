import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests

###############################################################################
# A) 调用 Ollama 的本地接口
###############################################################################
def call_local_llm(prompt: str) -> str:
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["response"]  # 添加这行返回成功响应
    except Exception as e:
        print(f"API请求失败: {str(e)}")
        return f"报告生成失败：{str(e)}"



###############################################################################
# B) 生成医学报告
###############################################################################
def generate_medical_report(disease, confidence):
    # 修改后的严格模板（添加分隔符要求）
    prompt = f"""请根据眼底检查结果生成医学报告。模板为
    ## AI医学报告

    ### 检查结果
    {disease}（置信度：{confidence}%）

    ### 病情描述

    ### AI建议
    
    """
    
    report = call_local_llm(prompt)
    return process_report(report) if report else "报告生成失败"  # 添加空值检查

def process_report(raw_report: str) -> str:
    """通过固定分隔符截取正式报告"""
    delimiter = "## AI医学报告"
    
    if delimiter in raw_report:
        # 找到分隔符位置（保留分隔符）
        start_index = raw_report.index(delimiter)
        return raw_report[start_index:]
    
    # 保底策略：返回原始报告（理论上不会触发）
    return raw_report


###############################################################################
# 1) 占位模型 & 类别标签
###############################################################################
model = None
CLASSES = [
    "类示例1", "类示例2", "类示例3", 
    "类示例4", "类示例5", "类示例6",
    "类示例7", "类示例8"
]

###############################################################################
# 2) 图像预处理
###############################################################################
def preprocess_image(image_left, image_right):
    transform = transforms.Compose([
        transforms.Resize((456, 456)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    if isinstance(image_left, np.ndarray):
        image_left = Image.fromarray(image_left)
    if isinstance(image_right, np.ndarray):
        image_right = Image.fromarray(image_right)

    image_left  = transform(image_left)
    image_right = transform(image_right)

    merged_image = torch.cat((image_left, image_right), dim=0)
    return merged_image.unsqueeze(0)


###############################################################################
# 3) 预测并生成医学报告
###############################################################################
def predict(image_left, image_right):
    input_tensor = preprocess_image(image_left, image_right)

    # -----------------------------
    # TODO: 这里替换为真实的模型推理过程
    # -----------------------------
    detected_diseases = ["糖尿病视网膜病变", "青光眼"]
    if not detected_diseases:
        detected_diseases = ["未检测到疾病"]

    disease_str = ", ".join(detected_diseases)
    medical_report = generate_medical_report(disease_str, 92)
    return disease_str, medical_report  # 直接返回原始报告内容


###############################################################################
# 4) 使用 Blocks 优化界面布局并自定义样式
###############################################################################
custom_css = """
/* 优化后的 CSS */
/* 标题样式 */
#title {
  text-align: center;
  font-size: 2.5em;
  margin: 0.5em auto;
  color: #2B5876;
  font-weight: 600;
  letter-spacing: 1px;
}

/* 副标题样式 */
#subtitle {
  text-align: center;
  font-size: 1em;
  color: #666;
  margin: 0 auto 2em;
  max-width: 80%;
  line-height: 1.5;
}

/* 布局优化 */
.row {
  gap: 1rem !important;  /* 减少列间距 */
}

/* 图像容器样式 */
.my-img {
  width: 300px;
  height: 300px;
  border: 3px solid #f0f0f0;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* 检测按钮优化 */
#detect-button {
  width: 150px;        
  margin: 1.5rem 0 0; /* 移除左右 margin */
  background: linear-gradient(45deg, #2B5876, #4e4376);
  color: white !important;
  font-weight: bold;
  transition: all 0.3s;
}

/* 调整报告框样式 */
#report-box {
  height: 65vh;           /* 使用视窗高度单位 */
  max-height: 700px;      /* 最大高度限制 */
  padding: 1.2rem;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  overflow-y: scroll !important;  /* 强制显示滚动条 */
  scrollbar-width: thin;  /* 更细的滚动条 */
}

/* 优化滚动条样式 */
#report-box::-webkit-scrollbar {
  width: 8px;
}

#report-box::-webkit-scrollbar-track {
  background: #f1f1f1; 
}

#report-box::-webkit-scrollbar-thumb {
  background: #888; 
  border-radius: 4px;
}

#report-box::-webkit-scrollbar-thumb:hover {
  background: #555; 
}

/* 优化Markdown显示 */
#report-box h3, #report-box h4 {
  color: #2B5876 !important;
  margin-top: 1em !important;
}

#report-box ul {
  padding-left: 1.5em;
  margin: 0.8em 0;
}

#report-box li {
  margin: 0.5em 0;
}
"""

with gr.Blocks(css=custom_css) as demo:
    # 标题与副标题（已居中）
    gr.Markdown("<p id='title'>👁️ AI 眼底检测系统</p>")
    gr.Markdown(
        "<p id='subtitle'>本系统基于深度学习模型分析左右眼眼底图像，通过本地部署的 deepseek-r1:1.5b 大模型生成医学报告<br>"
        "<em style='color: #e74c3c; font-size: 0.9em;'>检测结果仅供参考，实际诊断请咨询专业医生</em></p>"
    )

    # 紧凑布局
    with gr.Row(equal_height=True):
        # 左侧输入区
        with gr.Column(scale=1, min_width=340):
            left_eye_input = gr.Image(type="numpy", label="左眼图像", elem_classes="my-img")
            right_eye_input = gr.Image(type="numpy", label="右眼图像", elem_classes="my-img")
            with gr.Row(elem_classes="button-container"):  # 新增容器
                detect_button = gr.Button("开始检测", elem_id="detect-button")

        # 右侧输出区
        with gr.Column(scale=2, min_width=500):
            disease_output = gr.Textbox(label="检测结果", interactive=False, elem_id="disease-box")
            with gr.Group():
                gr.Markdown("")
                report_output = gr.Markdown(
                    elem_id="report-box",
                    value="等待生成报告...",
                )

    # 修改点击事件返回值
    detect_button.click(
        fn=predict,
        inputs=[left_eye_input, right_eye_input],
        outputs=[disease_output, report_output]
    )

if __name__ == "__main__":
    demo.launch()
