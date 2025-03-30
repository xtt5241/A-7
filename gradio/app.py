import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
import matplotlib.pyplot as plt
import pandas as pd
import os

import matplotlib
import matplotlib.font_manager as fm
# 1) 如果你想**动态**添加一款本地字体文件（如黑体SimHei）：
fm.fontManager.addfont(r"C:\Windows\Fonts\SimHei.ttf")  # 字体文件的绝对路径

# 2) 告诉 Matplotlib 这就是你要用的默认 sans-serif 字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

# 3) 解决负号“－”显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False
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
# todo 修改置信度,修改报告格式
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



# =================== 功能函数 ===================
# 路径
PREPROCESS_DIR = os.path.normpath("dataset/xxr/preprocess_images")
BASE_DIR = os.path.normpath("dataset/Training_Dataset")
EXCEL_DIR = os.path.normpath("dataset/training_annotation_(English).xlsx")


# 根据id获取原图片路径
# todo
def get_image_path_by_id(patient_id, eye):
    path= "dataset/Training_Dataset/"+patient_id + "_" + eye+ ".jpg"
    # print("原图片路径：",path)
    return path

# 根据id获取预处理后的图片路径
# todo
def get_preprocessed_image_path_by_id(patient_id, eye):
    path= "dataset/xxr/preprocess_images/"+patient_id + "_" + eye+ "_preprocess.jpg"
    # print("预处理后图片路径：",path)
    return path



# 根据上传的图片路径获取id
# todo
def get_id_by_uploaded_image_path(image_path):
    # 假设你有一个函数 get_id_by_image_path(image_path) 可以根据图片路径获取ID
    # 这里只是一个示例，你需要根据实际情况实现这个
    return "patient_id"

# =================== 单组导入的函数 ===================
import gradio as gr
from PIL import Image
import os

# 读取并返回预处理图像
# 1. 指定你预处理图像所在的目录（自动兼容 Windows 和 Linux）


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


# =================== 批量读取的函数 ===================
data2 = [[]]
def get_selected_id(df, evt: gr.SelectData):
    # evt.index = 被点击的行号
    if evt.index is None:
        return ""

    row_series = df.iloc[evt.index]  # 期望是一行
    patient_id_val = row_series["id"]

    # 如果不小心取到多行(或重复索引)就会是一个Series
    if isinstance(patient_id_val, pd.Series):
        # 只取第一个元素
        patient_id_val = patient_id_val.iloc[0]

    # 转成字符串返回给文本框
    return str(patient_id_val)

# 根据ID获取病人信息
def get_information_by_id(batch_df, patient_id):
    # 如果没有选择任何行，patient_id 可能为空
    if not patient_id:
        return "", "", "", "",None, None

    # 1) 用布尔索引或查询语句, 找到 DataFrame 中 id == patient_id 的行
    #    若 id 列原本是 int 类型，需要做一次 astype(str) 与点击后的 string 比较
    matched = batch_df[ batch_df["id"].astype(str) == str(patient_id) ]

    # 2) 如果找不到对应行，就返回一些提示或默认值
    if matched.empty:
        return str(patient_id), "未找到年龄", "未找到性别","未找到病症", None, None

    # 3) 否则取第一条匹配结果
    #    row 是一个 pd.Series，包含 "id", "age", "ill" 这几列
    row = matched.iloc[0]
    # print("row",row)

    # 从这行中提取年龄、性别、病症
    patient_age = row["年龄"]
    patient_sex = row["性别"]
    patient_ill = row["疾病"]

    # 4) 如果需要根据 ID 获取预处理图像
    #    这里只是演示，你要自行实现 get_preprocessed_image_path_by_id()
    left_image_path = get_preprocessed_image_path_by_id(str(patient_id), eye="left")   # 自定义实现
    right_image_path = get_preprocessed_image_path_by_id(str(patient_id), eye="right") # 自定义实现

    # 加载图像(若路径不存在或为空，则返回 None)
    left_image = Image.open(left_image_path) if left_image_path and os.path.exists(left_image_path) else None
    right_image = Image.open(right_image_path) if right_image_path and os.path.exists(right_image_path) else None

    # 5) 返回 6 个值，映射到前端的 6 个组件
    return (
        str(patient_id),     # 映射到 id_batch
        str(patient_age),    # 映射到 age_batch
        str(patient_sex),    # 映射到 sex_batch
        str(patient_ill),    # 映射到 ill_batch
        left_image,          # 映射到 left_pre_eye_output_batch
        right_image          # 映射到 right_pre_eye_output_batch
    )

# 数据分析部分
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def prepare_data_for_chart(data, col_index):
    """
    根据 col_index 判断要统计的是哪一列：
    - col_index == 3 => 疾病(需要拆分逗号)
    - col_index == 1 => 年龄(需要 pd.cut)
    - 其他 => 直接 value_counts()
    
    返回 (labels, sizes, title) 给后续画图函数使用
    """
    df = pd.DataFrame(data, columns=["id","年龄","性别","疾病"])

    if col_index == 3:
        # 处理疾病列
        all_ills = []
        for i in df.index:
            ill_str = df.loc[i, "疾病"]
            if not ill_str:
                continue
            ills = [x.strip() for x in ill_str.split(",") if x.strip()]
            all_ills.extend(ills)

        if not all_ills:
            # 没有有效疾病数据时，返回一个占位
            return ["无疾病数据"], [1], "疾病分布"

        series_data = pd.Series(all_ills).value_counts()
        labels = series_data.index.astype(str).tolist()
        sizes = series_data.values
        title = "疾病分布"
        return labels, sizes, title

    elif col_index == 1:
        # 年龄列 -> 分箱
        df["年龄"] = pd.to_numeric(df["年龄"], errors="coerce")
        bins = [0, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 200]
        labels_for_bins = [
            "0-17", "18-24", "25-29", "30-34", "35-39",
            "40-44", "45-49", "50-54", "55-59",
            "60-64", "65-69", "70+"
        ]
        df["age_range"] = pd.cut(df["年龄"], bins=bins, labels=labels_for_bins, right=False)
        series_data = df["age_range"].value_counts().sort_index()

        labels = series_data.index.astype(str).tolist()
        sizes = series_data.values
        title = "年龄分布"
        return labels, sizes, title

    else:
        # 其他列 -> value_counts()
        col_name = df.columns[col_index]
        series_data = df[col_name].value_counts()
        labels = series_data.index.astype(str).tolist()
        sizes = series_data.values
        title = f"{col_name}分布"
        return labels, sizes, title


def create_pie_chart_from_labels(labels, sizes, title="分布图"):
    """
    根据 labels、sizes 绘制饼图
    - 小于 5% 的扇区不显示数字
    - 右侧使用图例显示分类标签
    """
    # 如果需要中文显示，可自行指定中文字体
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False
    
    def autopct_func(pct):
        return ('%.1f%%' % pct) if pct >= 5 else ''

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=labels,             # 不在图中直接显示 label
        autopct=autopct_func,    # 小于5%不显示数字
        startangle=140,
        pctdistance=0.8,
        labeldistance=1.05,
    )
    ax.axis('equal')  # 保持饼图为圆形
    
    # 在右侧显示图例
    ax.legend(
        wedges, 
        labels, 
        title=title,
        loc="center left",
        bbox_to_anchor=(1.0, 0, 0.5, 1)  # 可微调位置
    )
    plt.tight_layout()
    return fig


def create_bar_chart(labels, sizes, title="分布图"):
    """
    根据 labels、sizes 绘制柱状图
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, sizes, color='steelblue')
    ax.set_xlabel("分类")
    ax.set_ylabel("数量")
    ax.set_title(title)
    plt.xticks(rotation=45)  # 如果分类很多，可旋转一下以免重叠
    plt.tight_layout()
    return fig


def plot_distribution(df_data, col_index, chart_type):
    """
    根据 chart_type 判断绘制饼图 or 柱状图
    - df_data: 来自前端存储的二维列表 (batch_information)
    - col_index: 统计哪一列 (1=年龄,3=疾病,...)
    - chart_type: "饼图" or "柱状图"
    """
    labels, sizes, title = prepare_data_for_chart(df_data, col_index)

    if chart_type == "柱状图":
        fig = create_bar_chart(labels, sizes, title)
    else:
        fig = create_pie_chart_from_labels(labels, sizes, title)
    
    return fig


def generate_analysis_report(data):
    """
    生成描述性统计报告：总数、平均年龄、男女人数、疾病分布等
    """
    df = pd.DataFrame(data, columns=["id","年龄","性别","疾病"])
    total_count = len(df)

    # 年龄
    df["年龄"] = pd.to_numeric(df["年龄"], errors="coerce")
    mean_age = df["年龄"].mean()
    median_age = df["年龄"].median()

    # 性别计数（注意你的Excel中是"Male"/"Female"或"M"/"F"要一致）
    male_count = (df["性别"] == "Male").sum()
    female_count = (df["性别"] == "Female").sum()

    # 拼装文字
    report = f"共导入样本：{total_count} 条记录\n\n"
    report += f"- 平均年龄：{mean_age:.2f}\n"
    report += f"- 中位数年龄：{median_age:.2f}\n"
    report += f"- 男性人数：{male_count}\n"
    report += f"- 女性人数：{female_count}\n\n"

    # 疾病分布
    all_ills = []
    for i in df.index:
        ill_str = df.loc[i, "疾病"]
        if not ill_str:
            continue
        ills = [x.strip() for x in ill_str.split(",") if x.strip()]
        all_ills.extend(ills)

    if all_ills:
        s = pd.Series(all_ills).value_counts()
        report += "**疾病分布：**\n"
        for disease_name, disease_count in s.items():
            report += f"- {disease_name}：{disease_count}例\n"
    else:
        report += "**疾病分布：**暂无有效疾病数据\n"

    return report


def upload_batch(excel_input):
    if not excel_input:
        # 如果用户没有上传文件或文件为空
        return []

    # 读取 Excel
    df = pd.read_excel(excel_input.name)

    # 结果列表，每行对应 ["id", "age", "sex","ill"]
    results = []

    # 需要检索的病症标签
    label_cols = ["N","D","G","C","A","H","M","O"]

    for i, row in df.iterrows():
        # 提取 ID, Patient Age
        patient_id = row.get("ID", "")
        patient_age = row.get("Patient Age", "")
        patient_sex = row.get("Patient Sex", "")

        # 收集值为1的病症列
        diseases = []
        for col in label_cols:
            if row.get(col, 0) == 1:
                diseases.append(col)

        # 用逗号拼接病症名称
        patient_ill = ", ".join(diseases)

        # 组合成单行结果
        results.append([patient_id, patient_age, patient_sex, patient_ill])

    # 返回二维列表
    return results


# 界面美化
###############################################################################
# H) 界面美化 - 自定义 CSS
###############################################################################
css = """
/* 外部容器，居中并限制整体宽度 */
#app-container {
  width: 100%;
  height: 100%;
}

/* 标题居中，大一点 */
#title {
  text-align: center;
  font-size: 2em;
  margin-bottom: 20px;
}

/* 副标题居中 */
#subtitle {
  text-align: center;
  margin-top: 10px;
  margin-bottom: 30px;
  font-size: 1.1em;
}

/* 让带有 center-button 类的按钮列居中 */
.center-button {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

.img_input {
  width: 100%;
  height: 100%;
  }


/* 疾病结果文本框的高度 */
#disease-box {
  min-height: 60px;
  width: 100%;
}

/* 报告区域增加高度并可滚动 */
#report-box {
  height: 723px;
  width: 100%;
  overflow: auto;
}


"""



# =================== Gradio 界面 ===================

# 创建 Gradio 界面
with gr.Blocks(css=css) as demo:
  # 标题
  gr.Markdown("<p id='title'>👁️ AI 眼底检测系统</p>")

# =================== Tab 2: 批量检测 ===================
  with gr.Tab(label="批量导入"):
    with gr.Row(elem_id="app-container"):
# 左侧输入区
      with gr.Column(scale=10):
        with gr.Row():
            excel_input = gr.File(label="上传Excel文件", file_types=[".xls", ".xlsx"])
            upload_batch_button = gr.Button(value="批量导入")
            # 点击批量按钮后，将预测结果更新到 Dataframe

# 列表显示病人信息
        with gr.Row():
            # 批量检测结果展示(表格)    
            batch_information = gr.Dataframe(
                value=data2,
                headers=["id","年龄","性别","疾病"],
                datatype=["str","number","str","str"],
                label="信息列表",
                # row_count=(5,"fixed"),   # 固定显示10行# 超出部分使用滚动条
                row_count=10,    # 固定显示10行# 超出部分使用滚动条
                col_count=(4,"fixed"),   # 固定显示10行# 超出部分使用滚动条
                # col_count=4,     # 固定显示10行# 超出部分使用滚动条
                wrap=True,
                interactive=True  # 一定要设为 True，才会触发 select
            )
            df_state = gr.State([])
            # 点击“批量导入”按钮后，将读取结果填充到 batch_information
            # 并同时更新 df_state
            def upload_and_store(excel_file):
                data_list = upload_batch(excel_file)  # 得到二维列表
                return data_list, data_list

            upload_batch_button.click(
                fn=upload_and_store,
                inputs=[excel_input],
                outputs=[batch_information, df_state]
            )

            upload_batch_button.click(
                fn=upload_batch,
                inputs=[excel_input],
                outputs=[batch_information]
        )

        with gr.Row():
            # 选择的行id
            selected_row_id = gr.Textbox(label="选中行的ID")
            batch_information.select(
                fn=get_selected_id,
                inputs=[batch_information],
                outputs=[selected_row_id]
            )
            # 点击按钮后，将选择的行号显示在 Textbox 中
            show_information_button = gr.Button("显示基本信息")

# 右侧输出区
      with gr.Column(scale=30):

        # 数据统计
        with gr.Tab(label="数据统计"):
            with gr.Row():
                analysis_btn = gr.Button("生成数据分析报告")
            with gr.Row():
                analysis_output = gr.Markdown(
                    elem_id="analysis-box",
                    value="等待生成分析报告...",
                )
            with gr.Row():
                # 用来指定统计的是疾病还是年龄列
                disease_col_index = gr.Number(value=3, visible=False)
                age_col_index = gr.Number(value=1, visible=False)

                # 用一个 Radio 让用户选“饼图”或“柱状图”
                chart_type_selector = gr.Radio(
                    ["柱状图", "饼图"],
                    value="柱状图",  # 默认选饼图
                    label="选择图表类型"
                )

            # 这里用两个按钮举例：分别统计“疾病”和“年龄”
            with gr.Row():
                disease_btn = gr.Button("查看疾病分布")
                age_btn = gr.Button("查看年龄分布")

            with gr.Row():
                out_plot = gr.Plot(label="图表输出")




            # 点击“疾病分布饼图” -> 统计第4列(索引3)
            # “查看疾病分布”按钮
            disease_btn.click(
                fn=plot_distribution,
                inputs=[df_state, disease_col_index, chart_type_selector], 
                outputs=out_plot
            )

            # “查看年龄分布”按钮
            age_btn.click(
                fn=plot_distribution,
                inputs=[df_state, age_col_index, chart_type_selector], 
                outputs=out_plot
            )


            analysis_btn.click(
                fn=generate_analysis_report,
                inputs=[df_state],
                outputs=[analysis_output]
            )

        # 基本信息
        with gr.Tab(label="基本信息"):
          with gr.Row():
            id_batch = gr.Textbox(label="ID")
            age_batch = gr.Textbox(label="年龄")
            sex_batch = gr.Textbox(label="性别")
          with gr.Tab(label="检测结果"):
              with gr.Row():
                left_pre_eye_output_batch = gr.Image(type="pil",label="左眼",width=500,height=500)
                right_pre_eye_output_batch = gr.Image(type="pil",label="右眼",width=500,height=500)
              with gr.Row():
                ill_batch = gr.Textbox(label="疾病类型",placeholder="疾病")

        show_information_button.click(
            fn=get_information_by_id,
            inputs=[batch_information,selected_row_id],
            outputs=[id_batch,age_batch,sex_batch,ill_batch,left_pre_eye_output_batch,right_pre_eye_output_batch]
        )
# =================== Tab 1: 单张检测 ===================
  with gr.Tab(label="单组导入"):
    with gr.Row(elem_id="app-container"):
      with gr.Column(scale=20,min_width=20):
      # 左侧输入区
        #   with gr.Column(scale=8,min_width=8):
        with gr.Row():
            left_eye_input = gr.Image(type="filepath", label="左眼图像",height=350,width=350)
            right_eye_input = gr.Image(type="filepath", label="右眼图像",height=350,width=350)
        # 上传图片按钮
        #   with gr.Column(scale=1,min_width=1, elem_classes="center-button"):  # 添加 elem_classes
        with gr.Row():
            upload_img_button =gr.Button("上传图像", elem_id="detect-button")
        # 预处理输出
        #   with gr.Column(scale=8,min_width=8):
        with gr.Row():
            left_pre_eye_output  = gr.Image(type="pil", label="左眼预处理后图像",height=350,width=350)
            right_pre_eye_output   = gr.Image(type="pil", label="右眼预处理后图像",height=350,width=350)

            # 上传图片按钮函数
            upload_img_button.click(
                fn=show_preprocessed_images,
                inputs=[left_eye_input, right_eye_input],
                outputs=[left_pre_eye_output, right_pre_eye_output]
            )


        # 预测按钮
        # todo 修改inputs和outputs
        #   with gr.Column(scale=1,min_width=1, elem_classes="center-button"):  # 添加 elem_classes
        with gr.Row():
            predict_button=gr.Button("开始预测", elem_id="detect-button")

      # 右侧输出区
      with gr.Column(scale=20,min_width=20):
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






# # =================== Tab 3: 病灶分割 ===================
#   with gr.Tab(label="病灶分割"):
#     with gr.Row(elem_id="app-container"):
# # 左侧输入区
#       with gr.Column(scale=5,min_width=5):
#           left_eye_input = gr.Image(type="numpy", label="左眼图像", min_width=40)
#           right_eye_input = gr.Image(type="numpy", label="右眼图像", min_width=40)
#       # 按钮
#       with gr.Column(scale=1,min_width=1, elem_classes="center-button"):  # 添加 elem_classes
#           spilit_button=gr.Button("病灶分割", elem_id="detect-button")
#           # spilit_button.click(
#           #     fn=predict,
#           #     inputs=[left_eye_input, right_eye_input],
#           #     outputs=[disease_output, report_output]
#           # )
# # 右侧输出区
#       # 血管分割
#       with gr.Column(scale=5,min_width=1):  # 添加 elem_classes
#             # 血管分割
#         with gr.Tab(label="血管分割"):
#             with gr.Column(scale=5,min_width=5):
#                 left_eye_input = gr.Image(type="numpy", label="左眼图像", min_width=40)
#                 right_eye_input = gr.Image(type="numpy", label="右眼图像", min_width=40)
#             # 视盘分割
#         with gr.Tab(label="视盘分割"):
#             with gr.Column(scale=20):
#                 left_eye_input = gr.Image(type="numpy", label="左眼图像", min_width=40)
#                 right_eye_input = gr.Image(type="numpy", label="右眼图像", min_width=40)
#             # 视杯分割
#         with gr.Tab(label="视杯分割"):
#             with gr.Column(scale=20):
#                 left_eye_input = gr.Image(type="numpy", label="左眼图像", min_width=40)
#                 right_eye_input = gr.Image(type="numpy", label="右眼图像", min_width=40)


 


  gr.Markdown(
      "<p id='subtitle'>本系统基于深度学习模型分析左右眼眼底图像，通过本地部署的 deepseek-r1:1.5b 大模型生成医学报告<br>"
      "<em style='color: #e74c3c; font-size: 0.9em;'>检测结果仅供参考，实际诊断请咨询专业医生</em></p>"
  )








if __name__ == "__main__":
    demo.launch()
