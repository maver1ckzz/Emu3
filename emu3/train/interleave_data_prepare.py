# -*- coding: utf-8 -*-

import argparse
import json
import os

from PIL import Image
import torch

from emu3.tokenizer import Emu3VisionVQModel, Emu3VisionVQImageProcessor


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='vision tokenizer path')
    parser.add_argument('--data-path', type=str, required=True, help='input json path')
    parser.add_argument('--output-path', type=str, required=True, help='tokenized data save path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for vision tokenizer')
    return parser.parse_args()


def smart_resize(image, image_area: int):
    w, h = image.size
    current_area = h * w
    target_ratio = (image_area / current_area) ** 0.5

    th = int(round(h * target_ratio))
    tw = int(round(w * target_ratio))

    image = image.resize((tw, th))
    return image


def encode_image(image_path, image_area, image_processor, image_tokenizer, device):
    image = Image.open(image_path).convert("RGB")
    image = smart_resize(image, image_area)

    # 关键：覆盖 processor 默认 min/max_pixels，避免被自动放大/缩小到别的尺寸
    image_processor.min_pixels = image_area
    image_processor.max_pixels = image_area
    image_processor.size = {
        "min_pixels": image_area,
        "max_pixels": image_area,
    }

    pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        pixel_values = pixel_values.to(device)
        token_ids = image_tokenizer.encode(pixel_values)

    token_ids = token_ids.squeeze(0).cpu().numpy()
    return token_ids


def main():
    args = prepare_args()

    image_processor = Emu3VisionVQImageProcessor.from_pretrained(args.model_path)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(args.model_path, device_map=args.device)
    image_tokenizer.eval()

    os.makedirs(f"{args.output_path}/feature", exist_ok=True)
    os.makedirs(f"{args.output_path}/list", exist_ok=True)

    datalist = {
        "prefix": f"{args.output_path}/feature",
        "path_list": []
    }

    with open(args.data_path, "r") as f:
        input_data = json.load(f)

    for inp in input_data:
        name = inp["name"]
        prompt = inp["text"]

        image_in_path = inp["image_in"]
        image_out_path = inp["image_out"]
        in_area = int(inp["in_area"])
        out_area = int(inp["out_area"])

        image_in_tokens = encode_image(
            image_path=image_in_path,
            image_area=in_area,
            image_processor=image_processor,
            image_tokenizer=image_tokenizer,
            device=args.device,
        )

        image_out_tokens = encode_image(
            image_path=image_out_path,
            image_area=out_area,
            image_processor=image_processor,
            image_tokenizer=image_tokenizer,
            device=args.device,
        )

        data = {
            "name": name,
            "image_in": image_in_tokens,
            "image_out": image_out_tokens,
            "texts": prompt,
        }

        torch.save(data, f"{args.output_path}/feature/{name}.pth")
        datalist["path_list"].append(f"{name}.pth")

    with open(f"{args.output_path}/list/train.json", "w") as f:
        json.dump(datalist, f)


if __name__ == "__main__":
    main()