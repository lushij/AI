import collections
import regex
import json
import os

class BPE_Tokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        # 初始词表：0-255 的 ASCII值
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.byte_encoder = {bytes([i]): i for i in range(256)}

        self.merges = {}
        self.next_token_id = 256
        self.cache = {}

        # --- 新增：特殊 Token 存储 ---
        # 结构: {"<|endoftext|>": 1001, ...}
        self.special_tokens = {}
        # 反向: {1001: "<|endoftext|>", ...}
        self.special_ids = {}

        # GPT-2 的标准正则
        self.GPT2_PAT = regex.compile(r"""
                        's|'t|'re|'ve|'m|'ll|'d       # 英语缩写
                        | [ ]?\p{L}+                  # 字母序列
                        | [ ]?\p{N}+                  # 数字序列
                        | [ ]?[^\s\p{L}\p{N}]+        # 标点符号
                        | \s+(?!\S)                   # 行尾空格
                        | \s+                         # 其他空格
                    """, regex.VERBOSE)

    def register_special_tokens(self, token_list):
        """
        手动注册特殊 Token，分配 ID
        通常在 train 结束后调用，或者 load 之后调用
        """
        for token_str in token_list:
            if token_str not in self.special_tokens:
                # 分配新 ID：也就是当前词表的长度
                new_id = len(self.vocab)

                # 存入词表 (为了 decode 时能还原)
                # 注意：特殊 token 作为字节串存入，方便统一处理
                self.vocab[new_id] = token_str.encode("utf-8")

                # 存入特殊映射表
                self.special_tokens[token_str] = new_id
                self.special_ids[new_id] = token_str

    def gpt_pre_tokenize(self, text):
        return self.GPT2_PAT.findall(text)

    def build_dict(self, words):
        word_freqs = collections.defaultdict(int)
        for word in words:
            word_bytes = tuple(word.encode("utf-8"))
            word_freqs[word_bytes] += 1
        return word_freqs

    def update_word_freqs(self, pair, new_token_id, word_freqs):
        new_word_freqs = collections.defaultdict(int)
        bigram = pair
        for word_tuple, freq in word_freqs.items():
            if bigram[0] not in word_tuple:
                new_word_freqs[word_tuple] += freq
                continue
            new_tuple = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == bigram[0] and word_tuple[i + 1] == bigram[1]:
                    new_tuple.append(new_token_id)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(new_tuple)] += freq
        return new_word_freqs

    def train(self, text, verbose=True):
        print(f"开始训练 BPE，目标词表大小: {self.vocab_size}")
        words = self.gpt_pre_tokenize(text)
        word_freqs = self.build_dict(words)

        # 留出空间给特殊 Token (例如预留 5 个位置)
        # 实际训练的 BPE 词表大小 = 目标大小 - 特殊Token数量(假设后续会加)
        # 这里为了简单，我们先训练满，然后追加特殊 Token，这会导致最终 vocab_size 略大于设定值，但无伤大雅

        while len(self.vocab) < self.vocab_size:
            pairs = collections.defaultdict(int)
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pairs[pair] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]

            new_token_id = self.next_token_id
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = new_token_id
            self.next_token_id += 1

            word_freqs = self.update_word_freqs(best_pair, new_token_id, word_freqs)

            # if verbose and len(self.vocab) % 100 == 0:
                # print(f"Merge {len(self.vocab)}: {best_pair} -> {new_token_id}")

        print(f"BPE 训练结束。当前词表大小: {len(self.vocab)}")

        # --- 训练结束后，自动注册常用的特殊 Token ---
        self.register_special_tokens(["<|endoftext|>", "<|padding|>"])
        print(f"特殊 Token 已添加。最终词表大小: {len(self.vocab)}")

        return self.vocab, self.merges

    def _encode_ordinary(self, text):
        """
        内部辅助函数：只处理普通文本的 BPE 编码 (就是你之前的 encode 逻辑)
        """
        words = self.gpt_pre_tokenize(text)
        ids = []
        for word in words:
            if word in self.cache:
                ids.extend(self.cache[word])
                continue

            word_ids = list(word.encode("utf-8"))
            while len(word_ids) >= 2:
                stats = {}
                for i in range(len(word_ids) - 1):
                    pair = (word_ids[i], word_ids[i + 1])
                    if pair in self.merges:
                        stats[pair] = self.merges[pair]
                if not stats:
                    break
                pair_to_merge = min(stats, key=stats.get)
                new_id = self.merges[pair_to_merge]
                new_ids = []
                i = 0
                while i < len(word_ids):
                    if i < len(word_ids) - 1 and word_ids[i] == pair_to_merge[0] and word_ids[i + 1] == pair_to_merge[1]:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(word_ids[i])
                        i += 1
                word_ids = new_ids

            self.cache[word] = word_ids
            ids.extend(word_ids)
        return ids

    def encode(self, text, allowed_special="all"):
        """
        对外暴露的 encode 接口：支持特殊 Token 处理
        allowed_special: "all" (解析特殊Token) 或 "none" (忽略特殊Token)
        """
        # 如果没有特殊 token 或者不允许处理，直接跑普通 BPE
        if allowed_special == "none" or not self.special_tokens:
            return self._encode_ordinary(text)

        # --- 核心逻辑：先切出特殊 Token ---
        # 构造正则模式：(<|endoftext|>|<|padding|>)
        # re.escape 用于转义字符（比如 | 符号）
        pattern = "(" + "|".join(regex.escape(k) for k in self.special_tokens) + ")"

        # 使用 split 切分，保留分隔符（即保留特殊 token）
        parts = regex.split(pattern, text)

        ids = []
        for part in parts:
            if not part: continue # 跳过空字符串

            if part in self.special_tokens:
                # 这是一个特殊 Token，直接查表拿 ID
                ids.append(self.special_tokens[part])
            else:
                # 这是一个普通文本，跑 BPE
                ids.extend(self._encode_ordinary(part))

        return ids

    def decode(self, ids):
        # 简单直接：直接拼 bytes
        # 因为 register_special_tokens 时，我们已经把特殊 token 作为 bytes 存入 vocab 了
        tokens = b"".join([self.vocab[idx] for idx in ids])
        return tokens.decode("utf-8", errors="replace")

    def save(self, file_prefix):
        merges_str = {f"{p[0]},{p[1]}": idx for p, idx in self.merges.items()}
        # 使用 latin-1 可以在 json 中安全存储任意 bytes
        vocab_str = {idx: token.decode('latin-1') for idx, token in self.vocab.items()}

        model_data = {
            "vocab": vocab_str,
            "merges": merges_str,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens # 保存特殊 token 定义
        }
        with open(f"{file_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        print(f"模型已保存到 {file_prefix}.json")

    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.merges = {}
        for k, v in data["merges"].items():
            p1, p2 = map(int, k.split(","))
            self.merges[(p1, p2)] = v

        self.vocab = {int(k): v.encode('latin-1') for k, v in data["vocab"].items()}
        self.next_token_id = max(self.vocab.keys()) + 1

        # 加载特殊 Token
        self.special_tokens = data.get("special_tokens", {})
        self.special_ids = {v: k for k, v in self.special_tokens.items()}

        print(f"模型加载完毕，包含特殊 Token: {list(self.special_tokens.keys())}")

def testBPE():
    # 1. 准备数据
    text = "Hello world! This is a test."

    # 2. 训练
    tokenizer = BPE_Tokenizer(vocab_size=300)
    tokenizer.train(text)  # 训练完会自动添加 <|endoftext|>

    # 3. 测试混合编码
    # 注意：这里我们手动在字符串里写了特殊 token
    input_text = "Hello world!<|endoftext|>This is padding:<|padding|>"

    print("\n--- 编码测试 ---")
    ids = tokenizer.encode(input_text)
    print(f"原文: {input_text}")
    print(f"编码 IDs: {ids}")

    # 验证 ID 是否正确
    eos_id = tokenizer.special_tokens["<|endoftext|>"]
    pad_id = tokenizer.special_tokens["<|padding|>"]
    print(f"其中 {eos_id} 是 EOS, {pad_id} 是 PAD")

    # 4. 解码测试
    decoded = tokenizer.decode(ids)
    print(f"解码回原文: {decoded}")

    # 5. 保存再加载
    tokenizer.save("llm_tokenizer")

    new_tok = BPE_Tokenizer()
    new_tok.load("llm_tokenizer.json")
    print(f"\n加载后特殊Token检查: {new_tok.special_tokens}")

def test_shakespeare_tokenizer():
        # 1. 读取刚才下载的莎士比亚全集
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()

        # 2. 训练分词器
        # 莎士比亚集比较小，vocab_size 设置为 3000 到 5000 就足够覆盖大部分词了
        tokenizer = BPE_Tokenizer(vocab_size=5000)
        tokenizer.train(text)

        # 3. 保存
        tokenizer.save("shakespeare_tokenizer")



# --- 完整流程测试 ---
if __name__ == "__main__":
    # test_shakespeare_tokenizer()
    new_tok = BPE_Tokenizer()
    new_tok.load("shakespeare_tokenizer.json")
    print(f"\n加载后特殊Token检查: {new_tok.special_tokens}")