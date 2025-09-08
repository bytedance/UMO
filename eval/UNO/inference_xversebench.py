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
import json
import os

from accelerate import Accelerator
from PIL import Image
from transformers import HfArgumentParser
from typing import Literal

from projects.UNO.uno.flux.pipeline import UNOPipeline, preprocess_ref


def image_grid(imgs, rows, cols):
    # assert len(imgs) == rows * cols

    w, h = imgs[0].size
    if imgs[0].mode == 'L':
        grid = Image.new('L', size=(cols * w, rows * h))
    else:
        grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

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
    
    with open(args.eval_json_path, "rt") as f:
        data_dicts = json.load(f)

    for i, data_dict in enumerate(data_dicts):
        if i % accelerator.num_processes != accelerator.process_index:
            continue

        index = data_dict["index"]

        ref_imgs = [
            Image.open(os.path.join("projects/XVerse", src_input["image_path"]))
            for src_input in data_dict["modulation"][0]["src_inputs"]
        ]
        if args.ref_size == -1:
            args.ref_size = 512 if len(ref_imgs) == 1 else 320

        ref_imgs = [preprocess_ref(img, args.ref_size) for img in ref_imgs]

        images_gen = []
        for j in range(args.num_images_per_prompt):
            image_gen = pipeline(
                prompt=data_dict["prompt"],
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                num_steps=args.num_steps,
                seed=args.seed + j,
                ref_imgs=ref_imgs,
                pe=args.pe,
            )
            images_gen.append(image_gen)

        image_gen = image_grid(images_gen, args.num_images_per_prompt // 2, 2)
        os.makedirs(args.save_path, exist_ok=True)
        prompt_name = data_dict["prompt"][:40].replace(" ", "_")
        image_gen.save(os.path.join(args.save_path, f"{index}_{prompt_name}.png"))

        # save config and image
        args_dict = vars(args)
        args_dict['prompt'] = data_dict["prompt"]
        args_dict['image_paths'] = data_dict["modulation"][0]["src_inputs"]
        with open(os.path.join(args.save_path, f"{index}_{prompt_name}.json"), 'w') as f:
            json.dump(args_dict, f, indent=4, ensure_ascii=False)        

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
