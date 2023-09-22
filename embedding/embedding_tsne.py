# 以下代码作用：
# 使用 t-SNE（t-Distributed Stochastic Neighbor Embedding）对具有高维嵌入的美食评论数据进行降维
# 并在二维空间中可视化这些数据点的分布
# 生成一个散点图，其中数据点的颜色表示评分等级，横坐标和纵坐标表示降维后的坐标

# 导入 pandas 包。Pandas 是一个用于数据处理和分析的 Python 库
# 提供了 DataFrame 数据结构，方便进行数据的读取、处理、分析等操作。
import pandas as pd
import ast
# 导入 NumPy 包，NumPy 是 Python 的一个开源数值计算扩展。这种工具可用来存储和处理大型矩阵，
# 比 Python 自身的嵌套列表（nested list structure)结构要高效的多。
import numpy as np
# 从 matplotlib 包中导入 pyplot 子库，并将其别名设置为 plt。
# matplotlib 是一个 Python 的 2D 绘图库，pyplot 是其子库，提供了一种类似 MATLAB 的绘图框架。
import matplotlib.pyplot as plt
import matplotlib
# 从 sklearn.manifold 模块中导入 TSNE 类。
# TSNE (t-Distributed Stochastic Neighbor Embedding) 是一种用于数据可视化的降维方法，尤其擅长处理高维数据的可视化。
# 它可以将高维度的数据映射到 2D 或 3D 的空间中，以便我们可以直观地观察和理解数据的结构。
from sklearn.manifold import TSNE

# 1 读取 fine_food_reviews_with_embeddings_1k 嵌入文件
output_datapath = "data/fine_food_reviews_with_embeddings_1k.csv"
df_embedded = pd.read_csv(output_datapath, index_col=0)
# 打印embedding的类型，结果为str
print(type(df_embedded["embedding"][0]))

# 1.1 将字符串类型转换为向量
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)
# # 打印embedding的类型，结果为pandas.core.series.Series
print(type(df_embedded["embedding_vec"]))

# 2 使用 t-SNE 可视化 1536 维 Embedding 美食评论

# 2.1 确保你的嵌入向量都是等长的
assert df_embedded['embedding_vec'].apply(len).nunique() == 1

# 2.2 将嵌入向量列表转换为二维 numpy 数组
matrix = np.vstack(df_embedded['embedding_vec'].values)

# 2.3 创建一个 t-SNE 模型，t-SNE 是一种非线性降维方法，常用于高维数据的可视化。
# n_components 表示降维后的维度（在这里是2D）
# perplexity 可以被理解为近邻的数量
# random_state 是随机数生成器的种子
# init 设置初始化方式
# learning_rate 是学习率。
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)

# 2.4 使用 t-SNE 对数据进行降维，得到每个数据点在新的2D空间中的坐标
vis_dims = tsne.fit_transform(matrix)

# 2.5 定义了五种不同的颜色，用于在可视化中表示不同的等级
colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]

# 2.6 从降维后的坐标中分别获取所有数据点的横坐标和纵坐标
x = [x for x, y in vis_dims]
y = [y for x, y in vis_dims]

# 2.7 根据数据点的评分（减1是因为评分是从1开始的，而颜色索引是从0开始的）获取对应的颜色索引
color_indices = df_embedded.Score.values - 1

# 2.8 确保你的数据点和颜色索引的数量匹配
assert len(vis_dims) == len(df_embedded.Score.values)

# 2.9 创建一个基于预定义颜色的颜色映射对象
colormap = matplotlib.colors.ListedColormap(colors)
# 2.10 使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定，alpha 是点的透明度
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)

# 2.11 为图形添加标题
plt.title("Amazon ratings visualized in language using t-SNE")

# 3 保存图形为图像文件（例如PNG格式）
plt.savefig("data/tsne_visualization.png")
# 显示图形
plt.show()

