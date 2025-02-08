import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import load_model, CLASSES
# from deepseek_chat import generate_medical_report
from grad_cam import get_grad_cam
import numpy as np
import cv2

# 1️⃣ 加载模型
model = load_model()

# 2️⃣ 图像预处理
from PIL import Image

def preprocess_image(image_left, image_right):
    transform = transforms.Compose([
        transforms.Resize((456, 456)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_left = transform(image_left)
    image_right = transform(image_right)

    # 🚀 拼接左右眼，变成 6 通道
    merged_image = torch.cat((image_left, image_right), dim=0)

    return merged_image.unsqueeze(0).to("cuda")



# 3️⃣ 预测疾病 + 生成医学报告 + Grad-CAM 可视化
def predict(image_left, image_right):
    # 🚀 确保输入是 PIL 图片
    if isinstance(image_left, np.ndarray):
        image_left = Image.fromarray(image_left)
    if isinstance(image_right, np.ndarray):
        image_right = Image.fromarray(image_right)

    image_tensor = preprocess_image(image_left, image_right)

    with torch.no_grad():
        output = model(image_tensor)  # 🚀 现在 output 是 8 个类别的概率
        prediction = (output.cpu().numpy() > 0.5).astype(int).flatten()  # 🚀 阈值 0.5

    detected_diseases = [CLASSES[i] for i in range(len(prediction)) if prediction[i] == 1]
    
    if not detected_diseases:
        detected_diseases = ["无疾病"]

    return ", ".join(detected_diseases)  # 返回预测类别字符串



# 4️⃣ Gradio 界面
iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="numpy"), gr.Image(type="numpy")],  # 🚀 传入 2 张图片
    outputs="text",  # 🚀 只返回类别，不返回 Grad-CAM
    title="👁 眼底疾病 AI 诊断系统",
    description="上传 **左右眼** 眼底图片，AI 识别疾病类别",
    examples=[["left_eye.jpg", "right_eye.jpg"]]
)



# 5️⃣ 启动 Web 应用
if __name__ == "__main__":
    iface.launch()
