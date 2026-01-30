"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/25
    Time:下午8:38
    To change this template use File | Settings | File Templates
"""
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
这里是手写TransformerLM里用到的组件，以及组装成完整的TransformerBlock
"""

class RMSNorm(nn.Module):
    """
    RMSNorm：均方根归一化层
    公式：x * (1 / RMS(x)) * gamma
    """

    def __init__(self, d_model=64, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 1. 这里的参数通常命名为 weight，对应公式里的 gamma
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x):
        # 2. 使用局部变量 ms，不要用 self.ms
        # .float() 是为了强制使用 FP32 进行统计量计算，防止数值溢出，非常好的细节！
        ms = x.pow(2).mean(dim=-1, keepdim=True).float()

        # 3. 计算倒数平方根 (1/RMS)
        rsqrt = torch.rsqrt(ms + self.eps)

        # 4. 这里的 type_as 是为了把数据转回原来的精度 (比如你是用半精度输入的)
        return x * rsqrt.type_as(x)

    def forward(self, x):
        # 5. 归一化后乘以缩放参数
        return self.weight * self._norm(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # 加上 bias=False，这是目前主流大模型的标配,因为后面有归一化，bais相当于没有了
        self.w_gate_value = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]

        # 1. 投影并切分，解包工业界常用
        # 输出维度是 d_ff * 2，所以这里直接解包(unpack)成两份
        gate, value = self.w_gate_value(x).chunk(2, dim=-1)

        # 2. SwiGLU 核心计算
        # Gate 过激活函数，Value 保持线性，然后相乘
        hidden = F.silu(gate) * value

        # 3. 输出投影
        return self.w_out(hidden)


class Rotate(nn.Module):
    def __init__(self, head_model, max_len):
        super().__init__()
        self.head_model = head_model
        self.max_len = max_len
        # 划分子空间
        subspace = torch.arange(0, head_model, 2)  # 0-d_k/2-1
        # 计算theta
        theta = torch.exp(-subspace * math.log(10000.0) / self.head_model).float()
        # 计算具体角度和位置有关
        pos = torch.arange(0, self.max_len).unsqueeze(1).float()

        angle = pos * theta  # shape [max_len,head_model/2]
        # 因为后面要进行拼接，所以这里要进行重复
        angle = torch.repeat_interleave(angle, 2, dim=-1)  # shape [max_len,head_model]
        # 正确点：求 angle 的余弦
        # register_buffer: 保证数据随模型去 GPU，且不被当做参数更新
        self.register_buffer('cos_cached', angle.cos().view(1, 1, max_len, head_model))
        self.register_buffer('sin_cached', angle.sin().view(1, 1, max_len, head_model))

    def _reverse_test(self, x):  # 这个只是用最简单的来演示一下
        # 输入尺寸为[batch,head,seq_len,head_model]
        # 先进行转置 将seq_len 和head_model交换达到转置的效果 ---> [batch,head,head_model,seq_len]
        x_t = x.transpose(-1, -2)
        # 创建一个容器来存放变换后的结果
        x_out_t = torch.zeros_like(x_t)
        # 将最后两维进行变换和加负号，最后两维形状[head_model,seq_len] 进行0行和1行交换，且0行加负号，2行和3行交换且2行加负号，以此类推
        for i in range(0, self.head_model, 2):
            x_out_t[:, :, i, :] = x_t[:, :, i + 1, :] * -1
            x_out_t[:, :, i + 1, :] = x_t[:, :, i, :]
        return x_out_t.transpose(-1, -2)  # 再转置回来

    def _reverse(self, x):
        # 输入尺寸为[batch,head,seq_len,head_model]
        # 把最后一维变成（组数 ，2）
        x = x.view(x.shape[:-1] + (-1, 2))  # 变成[batch heads seq_len head_model/2 2]
        # x.shape[:-1]只在最后一个维度进行变化
        # 拆解,因为最后一维是2，所以拆成两个张亮
        x1, x2 = x.unbind(dim=-1)
        # 旋转
        x_out = torch.stack((-x2, x1), dim=-1)
        # 再变回原来的形状
        return x_out.flatten(-2)

    def forward(self, x):
        # 获取当前输入的序列长度
        seq_len = x.shape[2]

        # 必须进行切片
        # 因为 x 可能比 max_len 短，所以只取前面 seq_len 长度的 cos/sin
        current_cos = self.cos_cached[:, :, :seq_len, :]
        current_sin = self.sin_cached[:, :, :seq_len, :]

        # 执行公式: x * cos + reverse(x) * sin
        return x * current_cos + self._reverse(x) * current_sin


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, rotatelayer: Rotate, dropout=0.01, max_len=1000):
        """
        :param d_model: 特征维度 (例如 512)
        :param n_heads: 头数 (例如 8)
        :param rotatelayer: 外部传入的旋转位置编码实例
        :param dropout: drop_out 概率
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        # 确保能整除
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.head_model = self.d_model // self.n_heads  # 每个头的维数

        # --- QKV 权重构造 ---
        # 注意：先定义 Linear，输入输出都是 d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # 因为后面要归一化，所以偏置值没有用
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # --- 输出投影层 ---
        # 合并头之后需要这一层来混合信息
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # --- 注入 RoPE ---
        self.rotate = rotatelayer

        # --- ★★★ 掩码实现 (Register Buffer) ★★★ ---
        # 创建一个下三角矩阵 (tril)，对角线及左下角为 1，右上角为 0
        # shape: [1, 1, max_len, max_len] 以便广播到 [batch, heads, ...]
        """
        # torch.ones: 造一个全 1 的方阵
        # torch.tril: (Triangle Lower) 只保留左下角，把右上角全变成 0
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))
        """
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                             .view(1, 1, max_len, max_len))  # 注册到gpu

    def forward(self, x):
        """
        :param x: [batch, seq_len, d_model]
        :return: [batch, seq_len, d_model]
        """
        # 获取尺寸
        batch, seq_len, _ = x.shape

        # --- 步骤 1: 线性投影 (Linear Projection) ---
        # 先把 d_model 维度的特征算出来
        # shape [batch, seq_len, d_model]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # --- 步骤 2: 拆分多头 (Split Heads) ---
        # view: [batch, seq_len, n_heads, head_model]
        # transpose: [batch, n_heads, seq_len, head_model] (把头放到前面，方便并行)
        q = q.view(batch, seq_len, self.n_heads, self.head_model).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_model).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_model).transpose(1, 2)

        # --- 步骤 3: 旋转位置编码 (RoPE) ---
        # 只对 Q 和 K 进行旋转，V 不动
        q = self.rotate(q)
        k = self.rotate(k)
        """
        #这是理解原理的写法，但是这样太吃显存
        # --- 步骤 4: 计算注意力分数 (Attention Scores) ---
        # Q @ K^T
        # q: [b, h, seq, d]
        # k.transpose: [b, h, d, seq]#后面那俩维度进行转置
        # result: [b, h, seq, seq]
        # 注意：这里要除以 sqrt(head_model)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_model)

        # --- 步骤 5: ★★★ 应用掩码 (Mask) ★★★ ---
        # 截取当前 seq_len 大小的 mask
        # 因为注意力分数是[b, h, seq, seq]
        mask = self.bias[:, :, :seq_len, :seq_len]  # 切出符合当前矩阵的尺寸，随着一句话的长度变化而变化
        # 将 mask 为 0 的位置 (未来信息) 填为负无穷 (-inf)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # --- 步骤 6: 归一化与加权 (Softmax & Value) ---
        # 在最后一维 (seq_len) 上做 softmax
        attn_weights = F.softmax(scores, dim=-1)  # 因为每一列都是我的不同对象，我要我看我和哪一个对象更紧凑，所以是-1那一维度
        attn_weights = self.dropout(attn_weights)

        # 加权求和: weights @ V
        # [b, h, seq, seq] @ [b, h, seq, d] -> [b, h, seq, d]
        out = torch.matmul(attn_weights, v)
        """
        #接下来优化性能
        # --- 🟢 新代码 (高性能版) ---
        # 这里的参数对应关系：
        # is_causal=True  <==> 对应你原来的 masked_fill(mask==0, -inf)
        # dropout_p=...   <==> 对应你原来的 self.dropout(...)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # 设为 None，因为我们用了 is_causal=True
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        # --- 步骤 7: 合并头 (Merge Heads) ---
        # transpose: [b, seq, h, d]
        # flatten/view: [b, seq, d_model]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len,
                                                    self.d_model)  # .contiguous()它申请了一块新内存，按照你现在的逻辑顺序，把书真正地搬了一次家。
        # .view() 必须要求内存连续。

        # 最后的输出线性层
        return self.w_o(out)




class TransformerBlock(nn.Module):
    def __init__(self,rmsnorm:RMSNorm,catt_mask_rotate:CausalSelfAttention,ffn:SwiGLU,d_model= 64,n_heads = 8,dropout = 0.01,max_len = 1000):
        """
        组合成一个完整的 Transformer Block
        :param rmsnorm: RMSNorm归一化层
        :param catt_mask_rotate: 多头注意力掩码旋转位置编码层
        :param ffn: SwiGLU前馈网络层
        :param d_model: 特征维数
        :param n_heads: 头数
        :param dropout: dropout概率
        :param max_len: 最大长度
        """
        super().__init__()
        self.rmsnorm = rmsnorm
        self.catt_mask_rotate = catt_mask_rotate
        self.ffn = ffn
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout=dropout
        self.max_len = max_len

    def forward(self, x):
        # x batch seq_len d_model
        x1 = self.rmsnorm(x)
        att_out = self.catt_mask_rotate(x1)
        x = x+att_out
        x2 = self.rmsnorm(x)
        ffn_out = self.ffn(x2)
        x = x+ffn_out
        return x


def test():
    d_model = 4
    rms = RMSNorm(d_model)
    x = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0]]  # 一个 Batch, 一个 Token, 4维特征
    ])
    out = rms(x)
    print("输入:", x)
    print("输出:", out)

    # 手算验证:
    # 平方: [1, 4, 9, 16] -> 均值: 30/4 = 7.5
    # RMS: sqrt(7.5) ≈ 2.7386
    # 归一化: [1, 2, 3, 4] / 2.7386 ≈ [0.365, 0.730, 1.095, 1.460]
    # 你的代码输出应该是这个！
def finally_test():
    if __name__ == "__main__":
        # --- 1. 设置超参数 ---
        BATCH_SIZE = 2
        SEQ_LEN = 10
        D_MODEL = 64  # 嵌入维度
        N_HEADS = 8  # 头数
        HEAD_DIM = D_MODEL // N_HEADS  # 8
        D_FF = 256  # FFN 中间层维度 (通常是 4倍 d_model 或 8/3倍)
        MAX_LEN = 100

        print("-" * 40)
        print("开始组装 Transformer Block...")

        # --- 2. 实例化子组件 ---
        # A. 旋转位置编码 (RoPE)
        rope = Rotate(head_model=HEAD_DIM, max_len=MAX_LEN)

        # B. 归一化 (RMSNorm)
        rms = RMSNorm(d_model=D_MODEL)

        # C. 注意力层 (Attention) - 注入 RoPE
        attn_layer = CausalSelfAttention(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            rotatelayer=rope,
            max_len=MAX_LEN
        )

        # D. 前馈网络 (SwiGLU)
        ffn_layer = SwiGLU(d_model=D_MODEL, d_ff=D_FF)

        # --- 3. 实例化完整的 Block ---
        block = TransformerBlock(
            rmsnorm=rms,
            catt_mask_rotate=attn_layer,
            ffn=ffn_layer
        )
        print("Transformer Block 组装完成！")

        # --- 4. 构造输入数据并测试 ---
        # 模拟输入: [Batch, Seq_Len, D_Model]
        x_input = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        print(f"输入尺寸: {x_input.shape}")

        # 前向传播
        x_output = block(x_input)

        print(f"输出尺寸: {x_output.shape}")
        print("-" * 40)

        # --- 5. 验证是否保持形状不变 ---
        if x_input.shape == x_output.shape:
            print("✅ 测试通过！输入输出尺寸一致。")
        else:
            print("❌ 测试失败！尺寸发生了变化。")

        # --- 6. 验证是否有梯度 (确保连接没断) ---
        loss = x_output.sum()
        loss.backward()
        print("✅ 反向传播通过！梯度正常计算。")


# --- 简单测试 ---
if __name__ == "__main__":
    finally_test()

