# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.append(os.path.abspath("projects/UNO"))

import dataclasses
import datasets
import json

from accelerate import Accelerator
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from typing import Literal

from projects.UNO.uno.flux.pipeline import UNOPipeline, preprocess_ref


class Collator:
    def __call__(self, features):
        return features

def horizontal_concat(images):
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_im

@dataclasses.dataclass
class InferenceArgs:
    eval_json_path: str | None = None
    offload: bool = False
    num_images_per_prompt: int = 1
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 512
    height: int = 512
    ref_size: int = -1
    num_steps: int = 25
    guidance: float = 4
    seed: int = 3407
    save_path: str = "output/inference"
    only_lora: bool = True
    lora_rank: int = 512
    lora_path: str | None = None
    pe: Literal['d', 'h', 'w', 'o'] = 'd'

def main(args: InferenceArgs):

    assert args.eval_json_path is not None, "Please provide eval_json_path"

    accelerator = Accelerator()

    pipeline = UNOPipeline(
        args.model_type,
        accelerator.device,
        args.offload,
        only_lora=args.only_lora,
        lora_rank=args.lora_rank,
    )
    if args.lora_path is not None:
        pipeline.load_ckpt(args.lora_path)
        pipeline.model.to(accelerator.device)
    
    if os.path.exists(args.eval_json_path):
        test_dataset = datasets.load_from_disk(args.eval_json_path)
    else:
        test_dataset = datasets.load_dataset(args.eval_json_path, split="train")
    loader = DataLoader(
        test_dataset,
        collate_fn=Collator(),
        batch_size=1,
        # shuffle=True,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    loader = accelerator.prepare(loader)

    for i, batched_data in enumerate(loader):
        data = batched_data[0]

        key = data['key']
        task_type = data['task_type']
        instruction = data['instruction']
        input_images = data['input_images']
        input_images = [ImageOps.exif_transpose(img) for img in input_images]

        if args.ref_size == -1:
            args.ref_size = 512 if len(input_images) == 1 else 320
        
        ref_imgs = [preprocess_ref(img, args.ref_size) for img in input_images]

        images_gen = []
        for j in range(args.num_images_per_prompt):
            image_gen = pipeline(
                prompt=instruction,
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                num_steps=args.num_steps,
                seed=args.seed + j,
                ref_imgs=ref_imgs,
                pe=args.pe,
            )
            images_gen.append(image_gen)
        
        sub_dir = os.path.join(args.save_path, "fullset", task_type)
        os.makedirs(sub_dir, exist_ok=True)
        output_image_path = os.path.join(sub_dir, f"{key}.png")

        if len(images_gen) > 1:
            for img_idx, image in enumerate(images_gen):
                image_name, ext = os.path.splitext(output_image_path)
                image.save(f"{image_name}_{img_idx}{ext}")
        
        output_image = horizontal_concat(images_gen)
        output_image.save(output_image_path)

        # save config and image
        args_dict = vars(args)
        args_dict['prompt'] = instruction
        with open(os.path.join(args.save_path, "fullset", task_type, f"{key}.json"), 'w') as f:
            json.dump(args_dict, f, indent=4, ensure_ascii=False)        

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
