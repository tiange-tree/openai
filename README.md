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

- 加载数据集
- 生成数据集的嵌入向量
- 使用 t-SNE 可视化低维 Embedding 美食评论

