# 以下代码作用：
# 使用 t-SNE（t-Distributed Stochastic Neighbor Embedding）对具有高维嵌入的美食评论数据进行降维
# 并在三维空间中可视化这些数据点的分布
# 生成一个散点图

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

# 2 创建一个3D图形对象

# 2.1 将嵌入向量列表转换为二维 numpy 数组
matrix = np.vstack(df_embedded['embedding_vec'].values)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 2.2 使用 t-SNE 对数据进行降维，将其从1536维降至3维
tsne = TSNE(n_components=3, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

# 2.3 定义评分值的范围，用于设置颜色映射的规范化
score_min = df_embedded.Score.min()
score_max = df_embedded.Score.max()
norm = matplotlib.colors.Normalize(vmin=score_min, vmax=score_max)

# 根据评分设置颜色映射
colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
colormap = matplotlib.colors.ListedColormap(colors)
colors = [colormap(norm(score)) for score in df_embedded.Score]

# 使用 t-SNE 降维后的坐标来绘制3D散点图，颜色和大小由评分决定
sc = ax.scatter(vis_dims[:, 0], vis_dims[:, 1], vis_dims[:, 2], c=colors, s=df_embedded.Score * 10, alpha=0.3)

# 添加坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 添加标题
plt.title("Amazon ratings visualized in 3D with color and size using t-SNE")

# 添加颜色映射图例，并指定要使用的 Axes 对象
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
colorbar = plt.colorbar(sm, label='Score', ax=ax)  # 将ax参数设置为ax，以指定Axes对象

# 3 保存图形为图像文件（例如PNG格式）
plt.savefig("data/tsne_3d_visualization.png")
# 显示图形
plt.show()
