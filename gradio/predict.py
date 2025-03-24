import pandas as pd

# 读取 Excel 文件并保存在全局变量中，避免反复读取
df = pd.read_excel("dataset/training_annotation_(English).xlsx")  # 你的 Excel 文件名

# 你在 Excel 中对应的疾病列
CLASSES = ["N","D","G","C","A","H","M","O"]

def get_diseases_from_excel(df, left_filename=None, right_filename=None):
    """
    在 df 中查找 'Left-Fundus' == left_filename 或 'Right-Fundus' == right_filename 的那一行，
    并根据列 N、D、G、C、A、H、M、O 的值(0/1)，生成疾病字符串。
    - 这里示例只根据 left_filename 查找，若需要右眼一样处理。
    """

    if left_filename is None:
        return "未检测到疾病"
    print("xtt_左眼",left_filename)
    # 在 df 中查找行
    # 注意：如果需要精确匹配左眼，就用下面这一句：
    row = df[df["Left-Fundus"] == left_filename]
    # 如果需要模糊匹配，就用下面这一句：
    # row = df[df["Left-Fundus"].str.contains(left_filename)]
    print("xtt_左眼_row",row)
    if row.empty:
        return "未检测到疾病"  # 如果没找到，可能是文件名不匹配

    # 我们只拿第一条匹配记录
    row = row.iloc[0]

    # 遍历 CLASSES 列，找出值 == 1 的列，对应疾病代码
    detected = []
    for disease_col in CLASSES:
        if disease_col in row:      # 确认列是否存在
            val = row[disease_col] # 这一列的数值
            if val == 1:
                detected.append(disease_col)

    if not detected:
        return "未检测到疾病"

    # 用逗号拼接
    return ",".join(detected)
