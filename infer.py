#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class SpineJSONLDataset(Dataset):
    def __init__(self, jsonl_path, size=224, aug=0):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        large_size = int(size * 256 / 224)

        # 推理时一般不用随机增强
        if aug == 0:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(large_size, interpolation=InterpolationMode.LANCZOS),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("RGB")
        img = self.transform(img)

        label = sample.get("label", None)
        if label is None:
            label = -1

        label = torch.tensor(label, dtype=torch.long)
        return img, label, idx


def ensure_label_shape(y):
    if isinstance(y, torch.Tensor):
        if y.ndim == 1:
            return y.unsqueeze(-1)
        return y
    elif isinstance(y, int):
        return torch.tensor([[y]], dtype=torch.long)
    else:
        raise ValueError(f"Unsupported label type: {type(y)}")


def build_model(args, out_dim):
    if args.model == "swin":
        model = timm.create_model("swin_base_patch4_window7_224", pretrained=False)
    elif args.model == "convnext":
        model = timm.create_model("convnextv2_base.fcmae", pretrained=False)
    elif args.model == "siglip":
        model = timm.create_model("vit_so400m_patch14_siglip_224.webli", pretrained=False, num_classes=3)
    elif args.model == "clip":
        model = timm.create_model("vit_base_patch16_clip_224.dfn2b", pretrained=False)
    elif args.model == "dinov3_conv":
        model = timm.create_model("convnext_base.dinov3_lvd1689m", pretrained=False)
    elif args.model == "dinov3_vit":
        model = timm.create_model("vit_base_patch16_dinov3_qkvb.lvd1689m", pretrained=False)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # 先加载预训练 backbone 权重（如果需要）
    if args.backbone_weights is not None and args.backbone_weights != "":
        if args.backbone_weights.endswith(".safetensors"):
            sd_raw = load_file(args.backbone_weights)
        else:
            sd_raw = torch.load(args.backbone_weights, map_location="cpu")

        sd_filt = {k: v for k, v in sd_raw.items() if k in model.state_dict()}
        model.load_state_dict(sd_filt, strict=False)
        print(f"Loaded backbone weights from: {args.backbone_weights}")

    model.reset_classifier(num_classes=out_dim)
    return model


# def load_trained_checkpoint(model, ckpt_path):
#     state_dict = torch.load(ckpt_path, map_location="cpu")

#     # 兼容 DataParallel / 非 DataParallel
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith("module."):
#             new_state_dict[k[len("module."):]] = v
#         else:
#             new_state_dict[k] = v

#     missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
#     print(f"Loaded trained checkpoint from: {ckpt_path}")
#     print(f"Missing keys: {len(missing)}")
#     print(f"Unexpected keys: {len(unexpected)}")
#     return model
def load_trained_checkpoint(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k

        # 先去掉 DataParallel 前缀
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]

        # 去掉你这个 checkpoint 的外层封装前缀
        if new_k.startswith("bb.0.model."):
            new_k = new_k[len("bb.0.model."):]

        # 保险一点，再兼容几种常见封装
        elif new_k.startswith("bb.0."):
            new_k = new_k[len("bb.0."):]
        elif new_k.startswith("model."):
            new_k = new_k[len("model."):]

        cleaned_state_dict[new_k] = v

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(cleaned_state_dict.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    matched = sorted(model_keys & ckpt_keys)

    print(f"matched: {len(matched)}")
    print(f"missing: {len(missing)}")
    print(f"unexpected: {len(unexpected)}")

    print("\n[missing sample]")
    for k in missing[:20]:
        print(k)

    print("\n[unexpected sample]")
    for k in unexpected[:20]:
        print(k)

    model.load_state_dict(cleaned_state_dict, strict=False)
    return model


def forward_pass(model, x, num_targets, num_classes):
    logits = model(x)  # (B, N*K)
    B = logits.shape[0]
    return logits.view(B, num_targets, num_classes)


def preds_from_logits(logits):
    return logits.argmax(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)

    parser.add_argument("--model", type=str, required=True,
                        choices=["swin", "convnext", "siglip", "clip", "dinov3_conv", "dinov3_vit"])
    parser.add_argument("--backbone_weights", type=str, default=None,
                        help="可选，backbone初始化权重，比如 .safetensors")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="训练好的分类头 checkpoint，比如 best.pth")

    parser.add_argument("--num_targets", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    # if args.model == "dinov3_vit":
    #     size = 256
    # else:
    size = 224

    dataset = SpineJSONLDataset(args.input_jsonl, size=size, aug=args.aug)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    out_dim = args.num_targets * args.num_classes
    model = build_model(args, out_dim)
    model = load_trained_checkpoint(model, args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    samples = dataset.samples

    with torch.no_grad():
        for x, y, indices in dataloader:
            x = x.to(device, non_blocking=True)
            logits = forward_pass(model, x, args.num_targets, args.num_classes)
            preds = preds_from_logits(logits).cpu().numpy()  # (B, N)

            for i, sample_idx in enumerate(indices.tolist()):
                pred = preds[i]

                if args.num_targets == 1:
                    samples[sample_idx]["predict"] = int(pred[0])
                else:
                    samples[sample_idx]["predict"] = pred.tolist()

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved prediction file to: {args.output_jsonl}")


if __name__ == "__main__":
    main()