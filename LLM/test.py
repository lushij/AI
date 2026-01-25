"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/25
    Time:下午7:29
    To change this template use File | Settings | File Templates
"""
import torch
import os

from BPE_Tokenizer import BPE_Tokenizer
from ManualEmbedding import ManualEmbedding, PositionEncoding # 优化了 import 写法
from TransformerBlock import *
d_model = 64
max_len = 1000
vocab_size = 1000

def test_embedding():
    # --- 1. 准备阶段 ---
    # 初始化分词器
    tokenizer = BPE_Tokenizer(vocab_size=1000)
    # 初始化 Embedding 层
    embedding_layer = ManualEmbedding(vocab_size=1000, d_model=64) #


    if os.path.exists("shakespeare_tokenizer.json"):
        tokenizer.load("shakespeare_tokenizer.json")
        print("已加载 shakespeare_tokenizer.json")
    else:
        print("未找到 tokenizer 文件，使用默认初始化参数运行测试...")

    # --- 2. 第一棒：Tokenizer ---
    text = "Hello world"
    token_ids_list = tokenizer.encode(text)
    print(f"1. Tokenizer 输出: {token_ids_list} (类型: {type(token_ids_list)})")

    # --- 3. 中间环节：数据打包 (List -> Tensor) ---
    input_tensor = torch.tensor(token_ids_list, dtype=torch.long).unsqueeze(0)
    print(f"2. 转换后的 Tensor: {input_tensor.shape} (形状: [B, T])")

    # --- 4. 第二棒：Embedding ---
    vector_output = embedding_layer(input_tensor)
    print(f"3. Embedding 输出: {vector_output.shape} (形状: [B, T, d_model])")

    return vector_output

def pos_and_embeding(vector_output):
    # 1. 初始化位置编码层
    pos_enc = PositionEncoding(d_model=d_model, max_len=max_len)

    # 2. 【关键】保留一份原始数据的副本，用于后续对比
    # .clone() 很重要，确保 original_data 不会随 vector_output 变化
    original_data = vector_output.clone()

    # 3. 执行位置编码叠加 (Add PE)
    output = pos_enc(vector_output)

    # 4. 打印形状
    print(f"\n--- 形状检查 ---")
    print(f"PE 矩阵形状: {pos_enc.pe.shape}")  # 应为[1000, 64]
    print(f"输出形状: {output.shape}")  # 应为[1, 4, 64] (取决于Hello world的分词长度)

    # 5. 【关键】验证数值变化
    print(f"\n--- 数值验证 (前5位小数) ---")
    # 取第一个样本，第一个token的前5个维度
    val_before = original_data[0, 0, :5].detach().numpy()
    val_after = output[0, 0, :5].detach().numpy()

    print(f"原始 Embedding: {val_before}")
    print(f"叠加 Position 后: {val_after}")

    # 6. 最终判定
    if not torch.equal(original_data, output):
        print("\n✅ 验证成功！数值已发生变化 (Embedding + PositionEncoding 生效)")
    else:
        print("\n❌ 验证失败！数值没有变化，请检查 PositionEncoding.forward() 是否写了 return x + self.pe")

if __name__ == '__main__':
    vector_output = test_embedding()
    pos_and_embeding(vector_output)