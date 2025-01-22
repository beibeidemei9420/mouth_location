import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = 'F:\ZED\YOLO\yolov8-zed-main\zed-yolo\conf_data.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 绘制每一列的散点图
for column in df.columns:
    plt.figure(figsize=(8, 6))  # 创建新的图形窗口

    if column=='0.5_distance':
        plt.scatter(df.index, df[column], c='#FF5718')
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,1.1])
        #plt.title(f'Scatter Plot of {column}')
        plt.xlabel('Frame')
        plt.ylabel(column)
        plt.grid(False)
        plt.savefig(column + '.png')
        plt.show()

    if column=='0.7_distance':
        plt.scatter(df.index, df[column])
        plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])
        #plt.title(f'Scatter Plot of {column}')
        plt.xlabel('Frame')
        plt.ylabel(column)
        plt.grid(False)
        plt.savefig(column + '.png')
        plt.show()

    if column=='0.9_distance':
        plt.scatter(df.index, df[column], c='green')
        #plt.title(f'Scatter Plot of {column}')
        plt.xlabel('Frame')
        plt.ylabel(column)
        plt.grid(False)
        plt.savefig(column + '.png')
        plt.show()
    if column=='0.8_distance':
        plt.scatter(df.index, df[column], c='#FF6664')
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
        plt.xlabel('Frame')
        plt.ylabel(column)
        plt.grid(False)
        plt.savefig(column + '.png')
        plt.show()
