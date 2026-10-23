import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

texts = [
     "特种作业人员必须持证上岗",          # A
    "电工、焊工需要取得操作资格证书",    # B：和 A 表述不同但含义相近
    "今天午饭吃什么好呢", 
    ]

response = client.embeddings.create(
    model="text-embedding-v3",
    input=texts,
)

vectors = [d.embedding for d in response.data]

def cos(a, b):
    """余弦相似度：把两个向量算出一个 -1~1 的分数，越接近 1 方向越一致、语义越相近"""
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    # a @ b 是向量点积；norm 是模长；点积/(两模长相乘) = 余弦相似度

print(f"向量维度: {len(vectors[0])}")         # text-embedding-v3 输出 1024 维
print(f"A-B 相似度: {cos(vectors[0], vectors[1]):.4f}")   # 预期较高
print(f"A-C 相似度: {cos(vectors[0], vectors[2]):.4f}")   # 预期明显更低
