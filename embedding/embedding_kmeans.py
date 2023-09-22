# 以下代码作用：
# 使用 K-Means 聚类，然后使用 t-SNE 可视化
# K-Means 聚类是一种常用的无监督学习算法
# 用于将数据点划分为不同的簇（clusters），使得同一簇内的数据点相似度较高，不同簇之间的数据点相似度较低。
# 它是一种迭代算法，其基本思想是将数据点分为 K 个簇，其中 K 是用户指定的参数

import pandas as pd
import ast
import numpy as np
# 从 scikit-learn中导入 KMeans 类。KMeans 是一个实现 K-Means 聚类算法的类。
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

# 1. 读取 fine_food_reviews_with_embeddings_1k 嵌入文件
output_datapath = "data/fine_food_reviews_with_embeddings_1k.csv"
df_embedded = pd.read_csv(output_datapath, index_col=0)
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)

# 将嵌入向量列表转换为二维 numpy 数组
assert df_embedded['embedding_vec'].apply(len).nunique() == 1
matrix = np.vstack(df_embedded['embedding_vec'].values)

# 2. 定义要生成的聚类数
n_clusters = 4

# 创建一个 KMeans 对象，用于进行 K-Means 聚类。
# n_clusters 参数指定了要创建的聚类的数量；
# init 参数指定了初始化方法（在这种情况下是 'k-means++'）；
# random_state 参数为随机数生成器设定了种子值，用于生成初始聚类中心。
# n_init=10 消除警告 'FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4'
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)

# 使用 matrix（我们之前创建的矩阵）来训练 KMeans 模型。这将执行 K-Means 聚类算法。
kmeans.fit(matrix)

# kmeans.labels_ 属性包含每个输入数据点所属的聚类的索引。
# 这里，我们创建一个新的 'Cluster' 列，在这个列中，每个数据点都被赋予其所属的聚类的标签。
df_embedded['Cluster'] = kmeans.labels_

# 3. 用t-SNE来可视化聚类
colors = ["red", "green", "blue", "purple"]

# 创建一个 t-SNE 模型，t-SNE 是一种非线性降维方法，常用于高维数据的可视化。
tsne_model = TSNE(n_components=2, random_state=42)
vis_data = tsne_model.fit_transform(matrix)

# 从降维后的数据中获取 x 和 y 坐标。
x = vis_data[:, 0]
y = vis_data[:, 1]

# 'Cluster' 列中的值将被用作颜色索引。
color_indices = df_embedded['Cluster'].values

# 创建一个基于预定义颜色的颜色映射对象
colormap = matplotlib.colors.ListedColormap(colors)

# 使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定
plt.scatter(x, y, c=color_indices, cmap=colormap)

# 为图形添加标题
plt.title("Clustering visualized in 2D using t-SNE")

# 4 保存图形为图像文件（例如PNG格式）
plt.savefig("data/kmeans_tsne_visualization.png")
# 显示图形
plt.show()
