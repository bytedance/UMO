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
sys.path.append(os.path.abspath("projects/OmniGen2"))

import dotenv

dotenv.load_dotenv(override=True)

import argparse
import json
import os
from typing import List, Tuple
from PIL import Image, ImageOps

import torch

from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from peft import LoraConfig
from safetensors.torch import load_file

from projects.OmniGen2.omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from projects.OmniGen2.omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OmniGen2 image generation script.")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name for output directory.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "dpmsolver"],
        help="Scheduler to use.",
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=50,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width."
    )
    parser.add_argument(
        "--max_input_image_pixels",
        type=int,
        default=1048576,
        help="Maximum number of pixels for each input image."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=5.0,
        help="Text guidance scale."
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=2.0,
        help="Image guidance scale."
    )
    parser.add_argument(
        "--cfg_range_start",
        type=float,
        default=0.0,
        help="Start of the CFG range."
    )
    parser.add_argument(
        "--cfg_range_end",
        type=float,
        default=1.0,
        help="End of the CFG range."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test data."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Path to save the generated images."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt."
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload."
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload."
    )
    parser.add_argument(
        "--enable_group_offload",
        action="store_true",
        help="Enable group offload."
    )
    parser.add_argument(
        "--disable_align_res",
        action="store_true",
        help="Align resolution to the input image resolution."
    )
    return parser.parse_args()
    
def load_pipeline(args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    from transformers import CLIPProcessor
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        processor=CLIPProcessor.from_pretrained(
            args.model_path,
            subfolder="processor",
            use_fast=True
        ),
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    if args.lora_path is not None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        lora_config = LoraConfig(
            r=512,
            lora_alpha=512,
            lora_dropout=0,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        pipeline.transformer.add_adapter(lora_config)
        lora_state_dict = load_file(args.lora_path, device=accelerator.device.__str__())
        pipeline.transformer.load_state_dict(lora_state_dict, strict=False)
        pipeline.transformer.fuse_lora(lora_scale=1, safe_fusing=False, adapter_names=["default"])
        pipeline.transformer.unload_lora()
    if args.scheduler == "dpmsolver":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(pipeline.transformer, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.mllm, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.vae, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:
        pipeline = pipeline.to(accelerator.device)
    return pipeline

def preprocess(input_image_path: List[str] = []) -> Tuple[str, str, List[Image.Image]]:
    """Preprocess the input images."""
    # Process input images
    input_images = None

    if input_image_path:
        input_images = []
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            input_images = [Image.open(os.path.join(input_image_path[0], f)).convert("RGB")
                          for f in os.listdir(input_image_path[0])]
        else:
            input_images = [Image.open(path).convert("RGB") for path in input_image_path]

        input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images

def run(args: argparse.Namespace, 
        accelerator: Accelerator, 
        pipeline: OmniGen2Pipeline, 
        instruction: str, 
        negative_prompt: str, 
        input_images: List[Image.Image]) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=args.width,
        height=args.height,
        align_res=not args.disable_align_res,
        num_inference_steps=args.num_inference_step,
        max_sequence_length=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",

    )
    return results

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

def main(args: argparse.Namespace, root_dir: str) -> None:
    """Main function to run the image generation process."""
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != 'fp32' else 'no')

    with open(args.test_data, "rt") as f:
        test_dataset = json.load(f)

    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.dtype == 'bf16':
        weight_dtype = torch.bfloat16

    # Load pipeline and process inputs
    pipeline = load_pipeline(args, accelerator, weight_dtype)

    for i, data_dict in enumerate(test_dataset):
        if i % accelerator.num_processes != accelerator.process_index:
            continue

        index = data_dict["index"]

        input_images = [
            Image.open(os.path.join("projects/XVerse", src_input["image_path"]))
            for src_input in data_dict["modulation"][0]["src_inputs"]
        ]
        instruction = data_dict["prompt"]

        results = run(args, accelerator, pipeline, instruction, args.negative_prompt, input_images)
        image_gen = image_grid(results.images, args.num_images_per_prompt // 2, 2)

        sub_dir = os.path.join(args.result_dir, args.model_name)
        os.makedirs(sub_dir, exist_ok=True)
        prompt_name = instruction[:40].replace(" ", "_")
        output_image_path = os.path.join(sub_dir, f"{index}_{prompt_name}.png")
        image_gen.save(output_image_path)

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)