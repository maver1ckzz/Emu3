import json
from types import SimpleNamespace

import torch

from emu3.mllm import Emu3Tokenizer
from emu3.train.datasets import Emu3FeatureDataset,Emu3InterleaveFeatureDataset

# 1. 构造和 train.py 一样的数据参数
args = SimpleNamespace(
    data_path="/mnt/nvme_share/wangty/emu3_data/c3_4_interleave_tokenized/list/train.json",
    null_prompt_prob=0.0,   # 调试时先关掉随机置空，方便看清楚
    apply_loss_on_only_vision=True,
    apply_loss_on_only_text=False,
    ignore_index=-100,
    visual_token_pattern="<|visual token {token_id:0>6d}|>",
    codebook_size=32768,
)

# 2. 加载 tokenizer
tokenizer = Emu3Tokenizer.from_pretrained(
    "/hdd/wangty/model/Emu3-Stage1",
    model_max_length=10240,
    padding_side="right",
    use_fast=False,
)

# 3. 构造 dataset
dataset = Emu3InterleaveFeatureDataset(args, tokenizer)

print("dataset length:", len(dataset))

# 4. 取一个样本
sample = dataset[0]

print("=" * 80)
print("sample keys:", sample.keys())
print("input_ids shape:", sample["input_ids"].shape)
print("attention_mask shape:", sample["attention_mask"].shape)
print("labels shape:", sample["labels"].shape)

# 5. 看前100个 token id
print("=" * 80)
print("first 100 input_ids:")
print(sample["input_ids"][:100])

print("=" * 80)
print("first 100 labels:")
print(sample["labels"][:100])

# 6. 解码前一段看看字符串内容
decoded = tokenizer.decode(sample["input_ids"][:300], skip_special_tokens=False)
print("=" * 80)
print("decoded prefix:")
print(decoded)

# 7. 看哪些位置参与 loss
valid_pos = (sample["labels"] != -100).nonzero(as_tuple=True)[0]
print("=" * 80)
print("num valid label positions:", len(valid_pos))
print("first 20 valid positions:", valid_pos[:20])

# 8. 如果想看这些位置对应的 token
print("=" * 80)
for pos in valid_pos[:20]:
    tid = sample["input_ids"][pos].item()
    tok = tokenizer.decode([tid], skip_special_tokens=False)
    print(pos.item(), tid, tok)