# LLM-From-Scratch：基于第一性原理的现代大模型架构复现

**项目简介：** 本项目旨在打破大语言模型（LLM）的“黑盒”迷思，完全摒弃高层封装库，基于 PyTorch 基础张量操作，从零构建了一个对标 LLaMA/PaLM 等前沿架构的生成式语言模型。通过纯手工实现每一层组件，深入剖析 Transformer 架构在训练与推理阶段的数学原理与数据流转过程。

**核心特性与技术栈：**

- **数据基石 (Tokenizer)**：
  - 手工实现 **BPE (Byte Pair Encoding)** 分词算法，不依赖 HuggingFace Tokenizers，完整复现从语料统计、词表构建到编解码的全流程，理解子词（Subword）切分的底层逻辑。
  - 构建自定义 **Embedding 层**，实现离散 Token 到连续向量空间的数学映射。
- **现代架构 (Modern Architecture)**：
  - **位置编码**：采用业界主流的 **RoPE (Rotary Positional Embeddings)** 旋转位置编码，通过矩阵运算简化，赋予模型更强的外推能力和相对位置感知力（优于传统的绝对位置编码）。
  - **归一化层**：摒弃传统的 LayerNorm，采用 **RMSNorm (Root Mean Square Layer Normalization)**，减少计算开销并提升数值稳定性，与 LLaMA 架构保持一致。
  - **前馈网络**：实现 **SwiGLU (Swish-Gated Linear Unit)** 激活函数，替代传统的 ReLU/GELU，利用门控机制显著提升模型的非线性表达能力和收敛速度。
- **训练优化 (Optimization)**：
  - 实现 **Warmup + Cosine Annealing (余弦退火)** 学习率调度策略，确保模型在训练初期稳定起步，后期精细收敛，避免陷入局部最优。
  - **手动实现梯度裁剪**，更有利于合适调整
  - **配置梯度累计**，是小显卡显示大显卡的威力
  - **混合精度加速 (AMP)**：集成 `torch.cuda.amp`，实现了 FP16 与 FP32 的混合精度训练。在保持模型收敛精度的同时，显著减少显存占用并提升计算吞吐量（Throughput）。

## 配置

```py
📊 模型总参数量: 56,897,024
💾 估算模型权重大小: 217.04 MB (FP32)
```

## 模型架构

![image-20260130110810606](\image_md\image-20260130110810606.png)

## 先看模型训练结果---预训练阶段

![image-20260130105912352](\image_md\image-20260130105912352.png)

## 再看模型微调阶段

![image-20260130110503753](\image_md\image-20260130110503753.png)



### 数据集采用的维基百科精简版（预训练）



```py
# 📄 文件名: download_wiki.py
import json
from datasets import load_dataset
from tqdm import tqdm

def prepare_wiki():
    print("⏳ 正在从 HuggingFace 下载维基百科精简版 (约500MB)...")
    # 自动下载并加载数据
    # 如果下载慢，可以开启镜像: export HF_ENDPOINT=https://hf-mirror.com
    dataset = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train")
    
    output_file = "wiki.txt"
    print(f"✅ 下载完成! 正在转换为纯文本: {output_file} ...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        # data['completion'] 里存的是正文
        for item in tqdm(dataset):
            text = item['completion']
            # 过滤掉太短的条目
            if len(text) > 100:
                f.write(text + "\n<|end|>\n") # 加上结束符
                
    print(f"🎉 搞定! 现在你可以直接运行 python train_pretrain.py 了")


prepare_wiki()
```

### 数据处理

```py
import re
import os
from tqdm import tqdm

class DataCleaner:
    def __init__(self):
        # 1. 定义什么算“标题” (特征：字数少，且不是以句号/叹号/问号结尾)
        # 这里的 20 是经验值，你可以根据实际数据调整
        self.header_pattern = re.compile(r'^.{1,20}$') 
        self.punctuation = ('.', '。', '!', '！', '?', '？', '"', '”', '…')

    def is_likely_header(self, line):
        """判断一行是否像是一个‘标题’"""
        line = line.strip()
        # 如果太长，肯定不是标题，是正文
        if len(line) > 25: 
            return False
        # 如果以标点符号结尾，说明是句子，不是标题
        if line.endswith(self.punctuation):
            return False
        # 如果是纯数字或极短，可能是列表项，算作正文保留
        if line.isdigit():
            return False
        return True

    def clean_text_block(self, raw_lines):
        """
        核心清洗逻辑：处理单个条目（从 <|end|> 到下一个 <|end|> 之间的内容）
        """
        cleaned_lines = []
        
        # 第一步：基础清洗（去空格、全角转半角）
        temp_lines = []
        for line in raw_lines:
            # 替换全角空格和不可见字符
            line = line.replace('\u3000', ' ').replace('\xa0', ' ').strip()
            if not line:
                continue # 扔掉空行
            if line == "<|end|>": # 暂时不处理结束符
                continue
            temp_lines.append(line)

        # 第二步：高级清洗（去空标题）
        # 我们使用 while 循环来灵活控制索引，方便向后看一行 (Lookahead)
        i = 0
        while i < len(temp_lines):
            current_line = temp_lines[i]
            
            # 判断当前行是不是标题
            if self.is_likely_header(current_line):
                # 看看下一行是否存在
                if i + 1 < len(temp_lines):
                    next_line = temp_lines[i+1]
                    # ★★★ 核心逻辑 ★★★
                    # 如果当前是标题，且下一行也是标题 -> 说明当前标题是空的 -> 删掉！
                    if self.is_likely_header(next_line):
                        i += 1
                        continue # 跳过当前行（即删除了空标题）
                else:
                    # 如果当前是标题，但已经是最后一行了 -> 说明是悬空标题 -> 删掉！
                    i += 1
                    continue

            cleaned_lines.append(current_line)
            i += 1

        # 如果清洗完只剩很少的内容（比如只剩一个标题），这数据也没用，直接丢弃
        if len(cleaned_lines) < 2:
            return None
            
        return "\n".join(cleaned_lines) + "\n<|end|>\n"

def process_file(input_path, output_path):
    cleaner = DataCleaner()
    
    if not os.path.exists(input_path):
        print(f"❌ 找不到输入文件: {input_path}")
        return

    print(f"🧹 开始清洗: {input_path} -> {output_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        buffer = []
        # 使用 tqdm 显示进度 (按字节估算)
        pbar = tqdm(total=os.path.getsize(input_path), unit='B', unit_scale=True)
        
        for line in f_in:
            pbar.update(len(line.encode('utf-8')))
            
            # 遇到结束符，说明凑齐了一个条目，开始清洗
            if "<|end|>" in line:
                if buffer:
                    cleaned_block = cleaner.clean_text_block(buffer)
                    if cleaned_block:
                        f_out.write(cleaned_block)
                buffer = [] # 清空缓冲区
            else:
                buffer.append(line)
        
        # 处理文件末尾可能残留的最后一段
        if buffer:
            cleaned_block = cleaner.clean_text_block(buffer)
            if cleaned_block:
                f_out.write(cleaned_block)
                
    print("\n✅ 数据清洗完成！")


# 配置你的文件路径
INPUT_FILE = "wiki.txt"          # 你的原始脏数据
OUTPUT_FILE = "wiki_clean.txt"   # 清洗后的干净数据

process_file(INPUT_FILE, OUTPUT_FILE)
```

### **数据集介绍：Alpaca-ZH-51k（微调阶段）**

本项目在 SFT（Supervised Fine-Tuning）阶段使用了 `alpaca_zh_51k` 数据集。这是一个开源的中文指令微调数据集，旨在提升大语言模型对中文用户指令的理解和响应能力。

- **数据来源**：该数据集基于斯坦福大学发布的 Alpaca 52k 数据集，通过机器翻译与人工校对相结合的方式迁移至中文语境，保留了原版多样化的任务类型（如代码生成、文本翻译、逻辑推理等）。
- **数据规模**：包含约 51,000 条样本。
- **数据格式**：每条样本由三部分组成：
  - `Instruction`（指令）：描述用户希望模型执行的任务。
  - `Input`（输入）：任务的上下文信息（可选）。
  - `Output`（输出）：模型预期的标准回答。
- **作用**：通过在该数据集上的训练，模型从单纯的“文本续写者”转变为能够理解人类意图的“AI 助手”。

## 训练结果

![image-20260130110657565](\image_md\image-20260130110657565.png)