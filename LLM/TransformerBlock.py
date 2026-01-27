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
这里是手写TransformerLM里用到的组件
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





























# --- 简单测试 ---
if __name__ == "__main__":
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

