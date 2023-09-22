# 导入 pandas 包。Pandas 是一个用于数据处理和分析的 Python 库
# 提供了 DataFrame 数据结构，方便进行数据的读取、处理、分析等操作。
import pandas as pd
# 导入 tiktoken 库。Tiktoken 是 OpenAI 开发的一个库，用于从模型生成的文本中计算 token 数量。
import tiktoken
# 从 openai.embeddings_utils 包中导入 get_embedding 函数。
# 这个函数可以获取 GPT-3 模型生成的嵌入向量。
# 嵌入向量是模型内部用于表示输入数据的一种形式。
from openai.embeddings_utils import get_embedding
import time
import openai

# 设定API密钥
openai.api_key = "YOUR_API_KEY"

# 1. 加载数据集「亚马逊美食评论数据集1000条」
input_datapath = "data/fine_food_reviews_1k.csv"
df = pd.read_csv(input_datapath, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()

# 将 "Summary" 和 "Text" 字段组合成新的字段 "combined"
df["combined"] = (
        "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)
df.head(2)
# print(df)

# 2. 生成数据集的嵌入向量
# 模型关键参数
# 模型类型，建议使用官方推荐的第二代嵌入模型：text-embedding-ada-002
embedding_model = "text-embedding-ada-002"
# text-embedding-ada-002 模型对应的分词器（TOKENIZER）
embedding_encoding = "cl100k_base"
# text-embedding-ada-002 模型支持的输入最大 Token 数是8191，向量维度 1536
# 在我们的 DEMO 中过滤 Token 超过 8000 的文本
max_tokens = 8000

# 2.1 将样本减少到最近的50个评论，并删除过长的样本
# 设置要筛选的评论数量为50，此值可以根据实际需求进行调整。
top_n = 1
# 对DataFrame进行排序，基于"Time"列，然后选取最后的2000条评论。
df = df.sort_values("Time").tail(top_n * 2)
# 丢弃"Time"列，因为我们在这个分析中不再需要它。
df.drop("Time", axis=1, inplace=True)

# 从'embedding_encoding'获取编码
encoding = tiktoken.get_encoding(embedding_encoding)
# 计算每条评论的token数量。我们通过使用encoding.encode方法获取每条评论的token数，然后把结果存储在新的'n_tokens'列中。
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))

# 如果评论的token数量超过最大允许的token数量，我们将忽略（删除）该评论。
# 我们使用.tail方法获取token数量在允许范围内的最后top_n（1000）条评论。
df = df[df.n_tokens <= max_tokens].tail(top_n)

# 打印出剩余评论的数量。
print(len(df))
# print(df)

# 2.2 生成嵌入向量
# 对df的combined每一列都应用get_embedding函数
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))


# 因openai api有调用限制，所以设置了每次调用api的间隔为5秒
# df["embedding"] = df.combined.apply(lambda x: get_embedding_with_interval(x,
# engine=embedding_model, interval_seconds=5))

# 定义一个函数，用于获取嵌入，并添加时间间隔
def get_embedding_with_interval(text, engine, interval_seconds=5):
    embedding = get_embedding(text, engine)  # 调用原始的 get_embedding 函数
    time.sleep(interval_seconds)  # 等待指定的时间间隔（秒）
    return embedding


# 2.3 将结果保存至csv中
output_datapath = "data/fine_food_reviews_with_embeddings_0.05k.csv"
df.to_csv(output_datapath)
