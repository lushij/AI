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

