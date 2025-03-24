import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
import matplotlib.pyplot as plt
import pandas as pd


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
model = "XTT/xxr_model/resnet50_dual_experiment_15_model_model6.pth"
CLASSES = ["N","D", "G", "C", "A", "H", "M", "O"]
# 读取 Excel 文件并保存在全局变量中，避免反复读取
df = pd.read_excel("dataset/training_annotation_(English).xlsx")  # 你的 Excel 文件名

from predict import get_diseases_from_excel
###############################################################################
# 3) 预测并生成医学报告
###############################################################################
def predict(left_eye_input, right_eye_input):
    # -----------------------------
    # TODO: 这里替换为真实的模型推理过程
    # -----------------------------

    # 提取上传图片的文件名
    left_fname = os.path.basename(left_eye_input)
    right_fname = os.path.basename(right_eye_input)
    # 拆分出文件名和扩展名
    left_basename, left_ext = os.path.splitext(left_fname)
    right_basename, right_ext = os.path.splitext(right_fname)
    # 加入扩展名".jpg"
    left_fundus=left_basename+'.jpg'
    right_fundus=right_basename+'.jpg'

    # print("左眼文件名",left_basename)
    detected_diseases = CLASSES
    if not detected_diseases:
        detected_diseases = ["未检测到疾病"]

    disease_str = get_diseases_from_excel(df, left_filename=left_fundus)
    print("疾病为:",disease_str)
    medical_report = generate_medical_report(disease_str, 92)
    return disease_str, medical_report  # 直接返回原始报告内容


# 示例数据
data = [['N', 30], ['D', 15], ['G', 25], ['C', 10], ['A', 20],['H', 10],['M', 10],['O', 10]]
data2 = [['xtt', '男','N'], ['xtt', '男','D'], ['xtt', '男','G'], ['xtt', '男','C'], ['xtt', '男','A'],['xtt', '男','H'],['xtt', '男','M'],['xtt', '男','O']]

def create_pie_chart(data):
    # 将输入的数据转换为 DataFrame
    df = pd.DataFrame(data, columns=['类别', '概率'])
    labels = df['类别']
    sizes = df['概率']
    
    # 创建饼状图
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # 确保饼图为圆形
    
    return fig

def create_pie_chart2(data):
    # 将输入的数据转换为 DataFrame
    df = pd.DataFrame(data, columns=['id', 'age'])
    labels = df['id']
    sizes = df['age']
    
    # 创建饼状图
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # 确保饼图为圆形
    
    return fig

# 批量读取excel文件
def batch_predict_by_excel(excel_input):
    # 读取 Excel 文件
    df = pd.read_excel(excel_input.name)
    # 初始化结果列表
    results = []
    # 遍历每一行数据

    return results

# 展示预处理结果
def show_preprocessing_on_click(excel_input):
    results = []

    return results

# =================== 读取并返回预处理图像 ===================
import gradio as gr
from PIL import Image
import os

# =================== 读取并返回预处理图像 ===================
import gradio as gr
from PIL import Image
import os

# 1. 指定你预处理图像所在的目录（自动兼容 Windows 和 Linux）
PREPROCESS_DIR = os.path.normpath("dataset/xxr/preprocess_images")

def show_preprocessed_images(left_path, right_path):
    """
    根据用户上传的左/右眼图像的原始路径，从预处理目录中读取对应的 _preprocess 文件并返回。
    
    参数:
        left_path (str): 左眼原图完整路径，例如 dataset/Training_Dataset/xxx.jpg
        right_path (str): 右眼原图完整路径

    返回:
        (Image, Image): 预处理后的左右眼图像（PIL.Image格式），用于Gradio显示
    """
    
    # 如果用户没有上传图片，或者路径为空，则直接返回 None
    if not left_path or not right_path:
        return None, None

    # 提取上传图片的文件名
    left_fname = os.path.basename(left_path)
    right_fname = os.path.basename(right_path)

    # 拆分出文件名和扩展名
    left_basename, left_ext = os.path.splitext(left_fname)
    right_basename, right_ext = os.path.splitext(right_fname)

    if (left_ext=='jpeg'):
        left_ext = '.jpg'
    right_ext = left_ext
    # print("左眼原图名：", left_basename)
    
    # 构造预处理后的文件名
    left_pre_fname = left_basename + "_preprocess" + '.jpg'
    right_pre_fname = right_basename + "_preprocess" + '.jpg'

    # print("预处理图名：", left_pre_fname, right_pre_fname)

    # 构造完整路径（使用 normpath 保证兼容性）
    left_pre_path = os.path.normpath(os.path.join(PREPROCESS_DIR, left_pre_fname))
    right_pre_path = os.path.normpath(os.path.join(PREPROCESS_DIR, right_pre_fname))

    # print("预处理路径：", left_pre_path, right_pre_path)

    # 读取图像（如不存在则为 None）
    left_pre_img = Image.open(left_pre_path) if os.path.exists(left_pre_path) else None
    right_pre_img = Image.open(right_pre_path) if os.path.exists(right_pre_path) else None
    # print("是否读取成功：", left_pre_img is not None, right_pre_img is not None)

    return left_pre_img, right_pre_img



# 创建 Gradio 界面
with gr.Blocks() as demo:
  # 标题
  gr.Markdown("<p id='title'>👁️ AI 眼底检测系统</p>")


# =================== Tab 1: 单张检测 ===================
  with gr.Tab(label="单组导入"):

    with gr.Row():
      # 左侧输入区
      with gr.Column(scale=5,min_width=5):
          left_eye_input = gr.Image(type="filepath", label="左眼图像", min_width=40)
          right_eye_input = gr.Image(type="filepath", label="右眼图像", min_width=40)
      # 上传图片按钮
      with gr.Column(scale=1,min_width=1, elem_classes="center-button"):  # 添加 elem_classes
          upload_img_button =gr.Button("上传图像", elem_id="detect-button")
      # 预处理输出
      with gr.Column(scale=5,min_width=5):
          left_pre_eye_output  = gr.Image(type="pil", label="左眼预处理后图像", min_width=40)
          right_pre_eye_output   = gr.Image(type="pil", label="右眼预处理后图像", min_width=40)

        # 上传图片按钮函数
          upload_img_button.click(
              fn=show_preprocessed_images,
              inputs=[left_eye_input, right_eye_input],
              outputs=[left_pre_eye_output, right_pre_eye_output]
          )


    # 预测按钮
    # todo 修改inputs和outputs
      with gr.Column(scale=1,min_width=1, elem_classes="center-button"):  # 添加 elem_classes
          predict_button=gr.Button("开始预测", elem_id="detect-button")

      # 右侧输出区
      with gr.Column(scale=20):
        disease_output = gr.Textbox(label="检测结果", interactive=False, elem_id="disease-box")
        with gr.Group():
            gr.Markdown("")
            report_output = gr.Markdown(
                elem_id="report-box",
                value="等待生成报告...",
            )
    # 预测按钮函数
        predict_button.click(
            fn=predict,
            inputs=[left_eye_input, right_eye_input],
            outputs=[disease_output, report_output]
        )


# =================== Tab 2: 批量检测 ===================
  with gr.Tab(label="批量导入"):
    with gr.Row():
      with gr.Column(5):
        excel_input = gr.File(label="上传Excel文件", file_types=[".xls", ".xlsx"])
        batch_button = gr.Button("开始批量检测")
        # 点击批量按钮后，将预测结果更新到 Dataframe
    #     batch_button.click(
    #         fn=batch_predict_by_excel,
    #         inputs=[excel_input],
    #         outputs=[batch_result]
    # )
# 列表显示病人信息
      with gr.Column(scale=10):
        # 批量检测结果展示(表格)
        batch_result = gr.Dataframe(
            value=data2,
            headers=["id","age","ill"], 
            datatype=["str","number","str"],
            label="信息列表",
            wrap=True,
            interactive=True  # 一定要设为 True，才会触发 select
        )
       # 核心：在 Dataframe 上注册 select 事件，
        # 事件回调函数 show_preprocessing_on_click 的第 2 个入参是 evt: gr.SelectData
        # 其中 evt.index 就是点击的行号
        # batch_result.select(
        #     fn=show_preprocessing_on_click,
        #     inputs=[excel_input],
        #     outputs=[batch_preprocess_plot]
        # )
# 可视化某行图像的预处理
      with gr.Column(scale=30):
        # 数据统计
        with gr.Tab(label="数据统计"):
          with gr.Row():
            plot_button2 = gr.Button("生成饼状图")
            plot_output2 = gr.Plot(label="饼状图")
            plot_button2.click(fn=create_pie_chart2, inputs=batch_result, outputs=plot_output2)
            # 统计图形
            gr.Plot(label="年龄段与疾病分布")
            gr.Plot(label="性别与疾病关联")
            gr.Plot(label="年龄段与疾病分布")
            gr.Plot(label="年龄段与疾病分布")
            gr.Plot(label="年龄段与疾病分布")
        # 基本信息
        with gr.Tab(label="基本信息"):
          with gr.Row():
            gr.Textbox(label="ID")
            gr.Textbox(label="Age")
          with gr.Tab(label="检测结果"):
              with gr.Row():
                gr.Image(label="左眼",)
                gr.Image(label="右眼")
              with gr.Row():
                gr.Textbox(label="疾病类型",placeholder="疾病")
          # AI报告
          with gr.Tab(label="AI报告"):
            with gr.Group():
                gr.Markdown("")
                report_output = gr.Markdown(
                    elem_id="report-box",
                    value="等待生成报告...",
                )

# =================== Tab 3: 病灶分割 ===================
  with gr.Tab(label="病灶分割"):
    with gr.Row():
# 左侧输入区
      with gr.Column(scale=5,min_width=5):
          left_eye_input = gr.Image(type="numpy", label="左眼图像", min_width=40)
          right_eye_input = gr.Image(type="numpy", label="右眼图像", min_width=40)
      # 按钮
      with gr.Column(scale=1,min_width=1, elem_classes="center-button"):  # 添加 elem_classes
          spilit_button=gr.Button("病灶分割", elem_id="detect-button")
          # spilit_button.click(
          #     fn=predict,
          #     inputs=[left_eye_input, right_eye_input],
          #     outputs=[disease_output, report_output]
          # )
# 右侧输出区
      # 血管分割
      with gr.Column(scale=5,min_width=1):  # 添加 elem_classes
            # 血管分割
        with gr.Tab(label="血管分割"):
            with gr.Column(scale=5,min_width=5):
                left_eye_input = gr.Image(type="numpy", label="左眼图像", min_width=40)
                right_eye_input = gr.Image(type="numpy", label="右眼图像", min_width=40)
            # 视盘分割
        with gr.Tab(label="视盘分割"):
            with gr.Column(scale=20):
                left_eye_input = gr.Image(type="numpy", label="左眼图像", min_width=40)
                right_eye_input = gr.Image(type="numpy", label="右眼图像", min_width=40)
            # 视杯分割
        with gr.Tab(label="视杯分割"):
            with gr.Column(scale=20):
                left_eye_input = gr.Image(type="numpy", label="左眼图像", min_width=40)
                right_eye_input = gr.Image(type="numpy", label="右眼图像", min_width=40)


 


  gr.Markdown(
      "<p id='subtitle'>本系统基于深度学习模型分析左右眼眼底图像，通过本地部署的 deepseek-r1:1.5b 大模型生成医学报告<br>"
      "<em style='color: #e74c3c; font-size: 0.9em;'>检测结果仅供参考，实际诊断请咨询专业医生</em></p>"
  )







if __name__ == "__main__":
    demo.launch()
