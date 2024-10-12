import pandas as pd

# 替换为你的 CSV 文件路径
data_file = 'dataset/Germany (DE).csv'

# 读取数据
data = pd.read_csv(data_file)

# 打印所有列名
print("所有列名:")
print(data.columns)

# 打印数据的详细信息
print("\n数据的详细信息:")
print(data.info())

# 打印前几行
print("\n前几行数据:")
print(data.head())