import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor
import torch
import json
from tqdm import tqdm
from emu3.mllm.processing_emu3 import Emu3Processor

with open("wty/data/c3_4_test.json", "r") as f:
    data = json.load(f)

EMU_HUB = "/hdd/wangty/diffuser_workdir/emu3/tra_c3-4/Emu3-Stage1-C34-Axial-256_8ep"
VQ_HUB = "/hdd/wangty/model/Emu3-VisionTokenizer"

model = AutoModelForCausalLM.from_pretrained(
    EMU_HUB,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

batch_size = 20
top_k=32
path = f"/hdd/wangty/diffuser_workdir/gen_img/tra_topk_{top_k}"
os.makedirs(path, exist_ok=True)

generation_config = GenerationConfig(
    use_cache=True,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
    max_new_tokens=4096,
    do_sample=True,
    top_k=32,
)

for start in tqdm(range(0, len(data), batch_size)):
    batch = data[start:start + batch_size]

    names = [item["name"] for item in batch]
    prompts = [item["text"] for item in batch]

    pos_inputs = processor(
        text=prompts,
        mode="G",
        ratio="1:1",
        image_area=65536,
        return_tensors="pt",
        padding="longest",
    )

    h = pos_inputs.image_size[:, 0]
    w = pos_inputs.image_size[:, 1]
    constrained_fn = processor.build_prefix_constrained_fn(h, w)

    logits_processor = LogitsProcessorList([
        PrefixConstrainedLogitsProcessor(
            constrained_fn,
            num_beams=1,
        ),
    ])

    with torch.inference_mode():
        outputs = model.generate(
            pos_inputs.input_ids.to("cuda:0"),
            generation_config=generation_config,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to("cuda:0"),
        )

    for j, name in enumerate(names):
        mm_list = processor.decode(outputs[j])
        for im in mm_list:
            if isinstance(im, Image.Image):
                im.save(f"{path}/{name}.png")