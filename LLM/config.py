"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/27
    Time:下午10:42
    To change this template use File | Settings | File Templates
"""
#这里是配置文件
"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/27
    Time:下午10:42
    To change this template use File | Settings | File Templates
"""



# --- 模型架构参数 ---
d_model = 512  # 嵌入维度
num_layers = 12  # Transformer Block 层数
num_heads = 8  # 多头注意力的头数
d_ff = 4*d_model  # 前馈网络中间层维度 (通常是 4*d_model)

# --- 序列与词表参数 ---
max_len = 6400  # 最大序列长度 (RoPE 和 Mask 需要用到)
vocab_size = 6405  # 词表大小 (对应 BPE Tokenizer 的词表)

# --- 正则化参数 ---
dropout = 0.1  # Dropout 概率
esp = 1e-6  # RMSNorm 的 epsilon 值

