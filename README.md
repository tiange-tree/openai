# openai使用指南

## 1. 设置环境变量

为了使用OpenAI API，你需要从OpenAI控制台获取一个API密钥。一旦你有了密钥，你可以将其设置为环境变量：

OpenAI链接：

```
https://platform.openai.com/account/api-keys
```

在MacOS的中断中运行以下命令：

```
export OPENAI_API_KEY='你的-api-key'
```

## 2. 安装依赖包

```
!pip install tiktoken openai pandas matplotlib plotly scikit-learn numpy
```

## 3. 项目内容

### 3.1 Embedding

- text_embedding.py 生成数据集的嵌入向量
- embedding_tsne.py 使用t-SNE对具有高维嵌入的美食评论数据进行降维，并生成二维可视化散点图
- embedding_tsne_3d.py 使用t-SNE对具有高维嵌入的美食评论数据进行降维，并生成三维可视化散点图
- embedding_kmeans.py 使用K-Means聚类对具有高维嵌入的美食评论数据进行聚类，并生成二维可视化散点图
- embedding_search_review.py 使用cosine_similarity函数来对嵌入向量进行相似度搜索，并生成相似度最高的3个评论

