# -*- coding: utf-8 -*-

import json
import os.path as osp
import random

import torch
from torch.utils.data import Dataset


class Emu3FeatureDataset(Dataset):

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__()

        self.args = args
        with open(args.data_path) as f:
            d = json.load(f)

        self.path_prefix = d["prefix"]
        self.filelist = d["path_list"]

        self.tokenizer = tokenizer
        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index: int):
        path = osp.join(self.path_prefix, self.filelist[index])
        data = torch.load(path)

        image_tokens = data["images"]
        image_prompt = self.format_image_prompt(image_tokens)

        p_prob = random.random()
        if p_prob < self.args.null_prompt_prob:
            prompt = ""
        else:
            prompt = data["texts"]

        input = self.tokenizer.bos_token + prompt + image_prompt
        sample = self.tokenizer(
            input,
            padding="max_length",
            return_token_type_ids=False,
            return_tensors="pt",
        )

        labels = sample["input_ids"]
        if self.args.apply_loss_on_only_vision:
            labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)
        # valid_token_count = (labels != self.args.ignore_index).sum().item()
        sample["labels"] = labels
        for k, v in sample.items():
            sample[k] = v.squeeze(0)

        return sample

    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token +
            f"{h}*{w}" +
            self.tokenizer.img_token +
            imgstr +
            self.tokenizer.eol_token +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return image_prompt

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr

class Emu3InterleaveFeatureDataset(Dataset):

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__()

        self.args = args
        with open(args.data_path) as f:
            d = json.load(f)

        self.path_prefix = d["prefix"]
        self.filelist = d["path_list"]

        self.tokenizer = tokenizer
        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]

        self.image_in_placeholder = "<image_in>"
        self.image_out_placeholder = "<image_out>"

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index: int):
        path = osp.join(self.path_prefix, self.filelist[index])
        data = torch.load(path)

        image_in_tokens = data["image_in"]
        image_out_tokens = data["image_out"]

        image_in_prompt = self.format_image_prompt(image_in_tokens)
        image_out_prompt = self.format_image_prompt(image_out_tokens)

        p_prob = random.random()
        if p_prob < self.args.null_prompt_prob:
            prompt = ""
        else:
            prompt = data["texts"]

        if self.image_in_placeholder not in prompt:
            raise ValueError(f"{self.image_in_placeholder} not found in text: {path}")
        if self.image_out_placeholder not in prompt:
            raise ValueError(f"{self.image_out_placeholder} not found in text: {path}")

        # 训练时完整替换：image_in 和 image_out 都替换为完整图像块
        prompt_filled = prompt.replace(self.image_in_placeholder, image_in_prompt)
        prompt_filled = prompt_filled.replace(self.image_out_placeholder, image_out_prompt)

        full_input = self.tokenizer.bos_token + prompt_filled

        sample = self.tokenizer(
            full_input,
            padding="max_length",
            return_token_type_ids=False,
            return_tensors="pt",
        )

        labels = sample["input_ids"].clone()

        if self.args.apply_loss_on_only_vision:
            labels = torch.where(
                torch.logical_and(labels >= self.bov, labels <= self.eov),
                labels,
                torch.full_like(labels, self.args.ignore_index)
            )

            # valid_token_count = (labels != self.args.ignore_index).sum().item()

            num_in_visual = image_in_tokens.shape[0] * image_in_tokens.shape[1]

            # labels 还是 [1, L]，所以取第二维才是 token 位置
            valid_pos = (labels != self.args.ignore_index).nonzero(as_tuple=True)[1]

            if valid_pos.shape[0] < num_in_visual:
                raise ValueError(
                    f"visual token 数量异常: {valid_pos.shape[0]} < image_in token 数 {num_in_visual}"
                )

            labels[0, valid_pos[:num_in_visual]] = self.args.ignore_index
        # valid_token_count = (labels != self.args.ignore_index).sum().item()

        sample["labels"] = labels
        for k, v in sample.items():
            sample[k] = v.squeeze(0)

        return sample
    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token +
            f"{h}*{w}" +
            self.tokenizer.img_token +
            imgstr +
            self.tokenizer.eol_token +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return image_prompt

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [
                self.args.visual_token_pattern.format(token_id=int(token_id))
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr
