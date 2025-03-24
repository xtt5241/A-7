# 总体

## 统计输出

- 疾病的年龄段的分布情况
- 男的女的年龄段
- 男生中最多出现的是什么
- 哪个年龄段最多出现的是哪类疾病

下面给你一份**修改后的示例代码**，演示如何：

1. **批量读取 Excel** 文件（假设含有 `id`, `age`, `ill`, `left_path`, `right_path` 等列）；  
2. **显示数据**到前端 `Dataframe`；  
3. **点击 Dataframe 行**时根据该行索引去加载左右眼图像并显示在界面上（不再需要用户手动上传）；  
4. 同时把对应的 `id、age、ill` 也显示到指定的 Textbox 或其他组件里。

以下示例代码中：

- `batch_predict_by_excel(excel_input)`：读取 Excel 并将其转成一个 DataFrame 返回给前端的 `batch_result` 表格。  
- `show_preprocessing_on_click(excel_input, evt)`：当用户点击表格某行时，通过 `evt.index` 找到对应行，从中读取 `left_path`, `right_path`, `id`, `age`, `ill` 等，**打开图像**后返回给前端的 `gr.Image` 组件，文本则返回给 `gr.Textbox`。  
- 由于你的原始代码中没有严格定义 Excel 里到底有哪些列，这里假设有五列：`id, age, ill, left_path, right_path`。你可以据实际需求改成别的名称。

你可将下述代码直接替换你已有的 `app.py` 里对应部分（主要修改了 “批量检测” 相关逻辑及输出组件）即可。

---

```python
import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
import matplotlib.pyplot as plt
import pandas as pd
import os

###############################################################################
# A) 调用本地大模型接口 (仅示例，可替换成真实接口)
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
        return response.json()["response"]
    except Exception as e:
        print(f"API请求失败: {str(e)}")
        return f"报告生成失败：{str(e)}"

###############################################################################
# B) 生成医学报告
###############################################################################
def generate_medical_report(disease, confidence):
    prompt = f"""请根据眼底检查结果生成医学报告。模板为
    ## AI医学报告

    ### 检查结果
    {disease}（置信度：{confidence}%）

    ### 病情描述

    ### AI建议
    
    """
    
    report = call_local_llm(prompt)
    return process_report(report) if report else "报告生成失败"

def process_report(raw_report: str) -> str:
    delimiter = "## AI医学报告"
    if delimiter in raw_report:
        start_index = raw_report.index(delimiter)
        return raw_report[start_index:]
    return raw_report

###############################################################################
# C) 模型推理示例
###############################################################################
def predict(image_left, image_right):
    # 这里的图像预处理 + 模型推理过程仅作示例
    # ----------------------------------------
    # 在你实际代码里，可直接调用自己的模型来返回疾病信息
    detected_diseases = ["糖尿病视网膜病变", "青光眼"]
    disease_str = ", ".join(detected_diseases)
    medical_report = generate_medical_report(disease_str, 92)
    return disease_str, medical_report

###############################################################################
# D) 示例：单张检测的数据与饼图
###############################################################################
data = [
    ['N', 30],
    ['D', 15],
    ['G', 25],
    ['C', 10],
    ['A', 20],
    ['H', 10],
    ['M', 10],
    ['O', 10]
]
def create_pie_chart(data):
    df = pd.DataFrame(data, columns=['类别', '概率'])
    labels = df['类别']
    sizes = df['概率']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    return fig

###############################################################################
# E) 批量导入：读取Excel + 在表格中显示 + 点击行显示结果
###############################################################################
def batch_predict_by_excel(excel_input):
    """
    读取Excel并返回DataFrame，用于批量检测结果展示。
    假设Excel里有这几列: id, age, ill, left_path, right_path
    """
    if excel_input is None:
        # 若没上传文件，就返回一个空表
        return pd.DataFrame(
            [["", "", "", "", ""]],
            columns=["id","age","ill","left_path","right_path"]
        )
    df = pd.read_excel(excel_input.name)

    # 这里可以自行做批量模型预测逻辑，比如：
    # for idx, row in df.iterrows():
    #     ... 读取路径做预测 ...
    #     df.at[idx, "ill"] = "预测结果"

    # 确保表格列名顺序与你前端Dataframe的headers一致
    return df[["id","age","ill","left_path","right_path"]]

def show_preprocessing_on_click(excel_input, evt: gr.SelectData):
    """
    点击 Dataframe 某行时的回调: 
    1) 根据选中行的索引 evt.index 获取对应的记录
    2) 读出 left_path, right_path, ill, id, age
    3) 打开图像并返回给前端 Image 组件
    4) 返回 id、age、ill 以显示到文本框
    """
    if excel_input is None:
        return None, None, "无ID", "无Age", "无疾病", "无报告"

    df = pd.read_excel(excel_input.name)
    if evt.index < 0 or evt.index >= len(df):
        return None, None, "无ID", "无Age", "无疾病", "无报告"

    row = df.iloc[evt.index]
    left_path = row.get("left_path", "")
    right_path = row.get("right_path", "")
    patient_id = str(row.get("id", "未知"))
    patient_age = str(row.get("age", "未知"))
    disease_str = str(row.get("ill", "未知"))

    # 如果真实有两张图，就读取它们
    left_img, right_img = None, None
    if os.path.exists(left_path):
        left_img = Image.open(left_path).convert("RGB")
    if os.path.exists(right_path):
        right_img = Image.open(right_path).convert("RGB")

    # 可选: 调用 generate_medical_report 或自己的模型再拿到AI报告
    # 这里只是示例
    medical_report = generate_medical_report(disease_str, 80)

    return left_img, right_img, patient_id, patient_age, disease_str, medical_report

###############################################################################
# F) 批量数据 2 的另一种饼图 (示例)
###############################################################################
def create_pie_chart2(data):
    # data 里 columns = ['id','age']
    df = pd.DataFrame(data, columns=['id', 'age'])
    labels = df['id']
    sizes = df['age']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    return fig

###############################################################################
# G) 构建 Gradio 界面
###############################################################################
custom_css = """
#title {
  text-align: center; 
  font-size: 2em; 
  margin-bottom: 10px;
}
"""

with gr.Blocks(css=custom_css) as demo:
    # 标题
    gr.Markdown("<p id='title'>👁️ AI 眼底检测系统</p>")

    # =========== Tab 1: 单张检测 ==============
    with gr.Tab(label="单组导入"):
        with gr.Row():
            # 左侧输入区
            with gr.Column(scale=5, min_width=5):
                left_eye_input = gr.Image(type="numpy", label="左眼图像", min_width=40)
                right_eye_input = gr.Image(type="numpy", label="右眼图像", min_width=40)

            # 按钮
            with gr.Column(scale=1, min_width=1, elem_classes="center-button"):
                detect_button = gr.Button("上传图像", elem_id="detect-button")

            # 预处理 (这里只是重复放两张图的示例，可以按需修改)
            with gr.Column(scale=5, min_width=5):
                left_eye_input2 = gr.Image(type="numpy", label="左眼图像2")
                right_eye_input2 = gr.Image(type="numpy", label="右眼图像2")

        with gr.Row():
            # 输出区
            disease_output = gr.Textbox(label="检测结果", interactive=False, elem_id="disease-box")
            report_output = gr.Markdown(value="等待生成报告...", elem_id="report-box")

        # 当点击按钮时调用 predict
        detect_button.click(
            fn=predict,
            inputs=[left_eye_input, right_eye_input],
            outputs=[disease_output, report_output]
        )

    # =========== Tab 2: 批量检测 ==============
    with gr.Tab(label="批量导入"):
        with gr.Row():
            with gr.Column(scale=5):
                excel_input = gr.File(label="上传Excel文件", file_types=[".xls", ".xlsx"])
                batch_button = gr.Button("开始批量检测")

                # 点击按钮后，将DataFrame返回给 batch_result
                # 假设Excel包含: id, age, ill, left_path, right_path
                batch_button.click(
                    fn=batch_predict_by_excel,
                    inputs=[excel_input],
                    outputs=[]
                )

            with gr.Column(scale=10):
                # 批量检测结果展示(表格)
                # 注意：这里的value不用再写死 data2，
                # 可以先给个空DataFrame或让它等待后端返回
                batch_result = gr.Dataframe(
                    headers=["id","age","ill","left_path","right_path"],
                    datatype=["str","number","str","str","str"],
                    label="信息列表",
                    wrap=True,
                    interactive=True
                )
                # 当按钮点击完毕后，把结果写到batch_result
                batch_button.click(
                    fn=batch_predict_by_excel,
                    inputs=[excel_input],
                    outputs=[batch_result]
                )

        # ========== 下方区域：点击表格行后显示的详细信息 ==========
        with gr.Row():
            with gr.Tab(label="数据统计"):
                # 示例：点击“生成饼状图”后，根据batch_result中的 id, age 画饼图
                plot_button2 = gr.Button("生成饼状图(仅ID/AGE演示)")
                plot_output2 = gr.Plot(label="饼状图")

                plot_button2.click(fn=create_pie_chart2, inputs=batch_result, outputs=plot_output2)

                gr.Markdown("此处可再放其他统计图: 如年龄段与疾病分布、性别与疾病关联等")

            with gr.Tab(label="基本信息"):
                # 这里放ID, Age, Ill之类的回显
                display_id   = gr.Textbox(label="ID")
                display_age  = gr.Textbox(label="Age")
                display_ill  = gr.Textbox(label="疾病类型")

                # 下面再做一个子Tab，显示左右眼及AI报告
                with gr.Tab(label="检测结果"):
                    left_eye_display = gr.Image(label="左眼")
                    right_eye_display = gr.Image(label="右眼")

                with gr.Tab(label="AI报告"):
                    report_output2 = gr.Markdown("等待生成报告...")

        # 当用户点击 batch_result 表格行时：加载图像 & 显示病人信息
        batch_result.select(
            fn=show_preprocessing_on_click,
            inputs=[excel_input],
            # outputs按顺序对应 return: left_img, right_img, id, age, ill, report
            outputs=[left_eye_display, right_eye_display,
                     display_id, display_age, display_ill, report_output2]
        )

    # 底部说明
    gr.Markdown(
      "<p id='subtitle'>本系统基于深度学习模型分析左右眼眼底图像，并可调用大模型生成医学报告<br>"
      "<em style='color: #e74c3c; font-size: 0.9em;'>检测结果仅供参考，实际诊断请咨询专业医生</em></p>"
    )

if __name__ == "__main__":
    demo.launch()
```

---

## 修改要点说明

1. **Excel 表格格式**  
   - 这里假设你的 Excel 有 5 列：`id, age, ill, left_path, right_path`。前两列存放病人信息，第三列是疾病，后两列是左右眼图像的本地路径。实际情况中可以换成你自己真实的列名，但必须和后端逻辑对应。

2. **批量检测**  
   - `batch_predict_by_excel(excel_input)` 在示例中只是读取 Excel 并原样返回，你可以在里面调用你的模型做批量预测，然后把预测结果写回到 DataFrame 的某一列（例如 `df["ill"] = xxx`），再返回给前端就行。

3. **点击表格行显示图像**  
   - 通过 `batch_result.select(...)` 来注册监听器，回调函数 `show_preprocessing_on_click` 会接收到一个 `gr.SelectData` 对象，其中 `evt.index` 是用户点击的行号。  
   - 拿到行号后，就能在 DataFrame 里取出 `left_path`, `right_path` 并 `Image.open(...)`，**return** 给 `gr.Image` 组件显示。  
   - 同理，把 `id`, `age`, `ill`、或 AI 生成的报告文本也 return 给 `gr.Textbox` 或 `gr.Markdown`。

4. **前端组件布局**  
   - 在 “数据统计” Tab 下，你放了一个按钮生成饼图；也留了一些占位的 `gr.Plot` 可以自行添加更多统计图。  
   - 在 “基本信息” Tab 下，你放几个 `Textbox` 用来显示 ID、Age、Disease；再嵌一个 “检测结果” Tab 给左右眼图像，一个 “AI报告” Tab 给报告文本。布局方式依你喜好定制即可。

5. **不再需要再次手动上传**  
   - 因为我们在 `show_preprocessing_on_click` 里直接用本地的图像路径进行 `Image.open`，返回 `PIL.Image` 给前端，`gr.Image` 会自动显示图像。  
   - 用户只需要点选表格某行，就能看到对应的图像，而不是再手动 `gr.Image` 上传。

这样即可实现“批量导入 → 点击某行查看左右眼图像、ID、Age、疾病、AI报告”等功能。你只需根据你的实际 Excel 列名和真实推理流程稍微调整即可。祝你开发顺利!
