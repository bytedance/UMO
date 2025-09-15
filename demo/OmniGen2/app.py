# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) VectorSpaceLab and its affiliates. All rights reserved.

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

import gradio as gr

import argparse
import json
import random
from datetime import datetime
from glob import glob
from typing import Literal

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from safetensors.torch import load_file

from projects.OmniGen2.omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from projects.OmniGen2.omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from projects.OmniGen2.omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from projects.OmniGen2.omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from projects.OmniGen2.omnigen2.utils.img_util import create_collage

NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
SAVE_DIR = "output/gradio"

pipeline = None
accelerator = None
save_images = False
enable_taylorseer = False
enable_teacache = False

def load_pipeline(accelerator, weight_dtype, args):
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    
    lora_path = hf_hub_download("bytedance-research/UMO", "UMO_OmniGen2.safetensors") if args.lora_path is None else args.lora_path
    target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=512,
        lora_alpha=512,
        lora_dropout=0,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    pipeline.transformer.add_adapter(lora_config)
    lora_state_dict = load_file(lora_path, device=accelerator.device.__str__())
    pipeline.transformer.load_state_dict(lora_state_dict, strict=False)
    pipeline.transformer.fuse_lora(lora_scale=1, safe_fusing=False, adapter_names=["default"])
    pipeline.transformer.unload_lora()
    
    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(accelerator.device)
    return pipeline


def run(
    instruction,
    width_input,
    height_input,
    image_input_1,
    image_input_2,
    image_input_3,
    scheduler: Literal["euler", "dpmsolver++"] = "euler",
    num_inference_steps: int = 50,
    negative_prompt: str = NEGATIVE_PROMPT,
    guidance_scale_input: float = 5.0,
    img_guidance_scale_input: float = 2.0,
    cfg_range_start: float = 0.0,
    cfg_range_end: float = 1.0,
    num_images_per_prompt: int = 1,
    max_input_image_side_length: int = 2048,
    max_pixels: int = 1024 * 1024,
    seed_input: int = -1,
    align_res: bool = True,
):
    if enable_taylorseer:
        pipeline.enable_taylorseer = True
    elif enable_teacache:
        pipeline.transformer.enable_teacache = True
        pipeline.transformer.teacache_rel_l1_thresh = 0.05

    input_images = [image_input_1, image_input_2, image_input_3]
    input_images = [img for img in input_images if img is not None]

    if len(input_images) == 0:
        input_images = None

    if seed_input == -1:
        seed_input = random.randint(0, 2**16 - 1)

    # generator = torch.Generator(device=accelerator.device).manual_seed(seed_input)
    generator = torch.Generator(device="cpu").manual_seed(seed_input) # set random to cpu to avoid different result on different GPU

    if scheduler == 'euler' and not isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler):
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
    elif scheduler == 'dpmsolver++' and not isinstance(pipeline.scheduler, DPMSolverMultistepScheduler):
        pipeline.scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=width_input,
        height=height_input,
        align_res=align_res,
        max_input_image_side_length=max_input_image_side_length,
        max_pixels=max_pixels,
        num_inference_steps=num_inference_steps,
        max_sequence_length=1024,
        text_guidance_scale=guidance_scale_input,
        image_guidance_scale=img_guidance_scale_input,
        cfg_range=(cfg_range_start, cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )

    vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
    output_image = create_collage(vis_images)

    output_path = ""
    if save_images:
        # Create outputs directory if it doesn't exist
        output_dir = SAVE_DIR
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Generate unique filename with timestamp
        output_path = os.path.join(output_dir, f"{timestamp}_seed{seed_input}_{instruction[:20]}.png")
        # Save the image
        output_image.save(output_path)

        # Save All Generated Images
        if len(results.images) > 1:
            for i, image in enumerate(results.images):
                image_name, ext = os.path.splitext(output_path)
                image.save(f"{image_name}_{i}{ext}")
    return output_image, output_path


def get_examples(base_dir="assets/examples/OmniGen2"):
    example_keys = ["instruction", "width_input", "height_input", "image_input_1", "image_input_2", "image_input_3", "seed_input", "align_res", "output_image", "output_image_OmniGen2"]
    examples = []
    example_configs = glob(os.path.join(base_dir, "*", "config.json"))
    for config_path in example_configs:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        _example = [config.get(k, None) for k in example_keys]
        examples.append(_example)
    return examples


with open("assets/logo.svg", "r", encoding="utf-8") as svg_file:
    logo_content = svg_file.read()
title = f"""
<div style="display: flex; align-items: center; justify-content: center;">
    <span style="transform: scale(0.7);margin-right: -5px;">{logo_content}</span>  
    <span style="font-size: 1.8em;margin-left: -10px;font-weight: bold; font-family: Gill Sans;">UMO (based on OmniGen2) by UXO Team</span>
</div>
""".strip()

badges_text = r"""
<div style="text-align: center; display: flex; justify-content: center; gap: 5px;">
<a href="https://github.com/bytedance/UMO"><img alt="Build" src="https://img.shields.io/github/stars/bytedance/UMO"></a> 
<a href="https://bytedance.github.io/UMO/"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-UMO-blue"></a> 
<a href="https://huggingface.co/bytedance-research/UMO"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=green"></a>
<a href="https://arxiv.org/abs/2509.06818"><img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-UMO-b31b1b.svg"></a>
<a href="https://huggingface.co/spaces/bytedance-research/UMO_UNO"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Demo&message=UMO-UNO&color=orange"></a>
<a href="https://huggingface.co/spaces/bytedance-research/UMO_OmniGen2"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Demo&message=UMO-OmniGen2&color=orange"></a>
</div>
""".strip()

tips = """
📌 ***UMO*** is a **U**nified **M**ulti-identity **O**ptimization framework to *boost the multi-ID fidelity and mitigate confusion* for image customization model, and the latest addition to the UXO family (<a href='https://github.com/bytedance/UMO' target='_blank'> UMO</a>, <a href='https://github.com/bytedance/USO' target='_blank'> USO</a> and <a href='https://github.com/bytedance/UNO' target='_blank'> UNO</a>).

🎨 UMO in the demo is trained based on <a href='https://github.com/VectorSpaceLab/OmniGen2' target='_blank'> OmniGen2</a>.

💡 We provide step-by-step instructions in our <a href='https://github.com/bytedance/UMO' target='_blank'> Github Repo</a>. Additionally, try the examples and comparison provided below the demo to quickly get familiar with UMO and spark your creativity!

<details>
<summary style="cursor: pointer; color: #d34c0e; font-weight: 500;"> ⚡️ Tips from the based OmniGen2</summary>

- Image Quality: Use high-resolution images (**at least 512x512 recommended**).
- Be Specific: Instead of "Add bird to desk", try "Add the bird from image 1 to the desk in image 2".
- Use English: English prompts currently yield better results.
- Increase image_guidance_scale for better consistency with the reference image:
    - Image Editing: 1.3 - 2.0
    - In-context Generation: 2.0 - 3.0
- For in-context edit (edit based multiple images), we recommend using the following prompt format: "Edit the first image: add/replace (the [object] with) the [object] from the second image. [descripton for your target image]."
- For example: "Edit the first image: add the man from the second image. The man is talking with a woman in the kitchen"
""".strip()

article = """
```bibtex
@article{cheng2025umo,
  title={UMO: Scaling Multi-Identity Consistency for Image Customization via Matching Reward},
  author={Cheng, Yufeng and Wu, Wenxu and Wu, Shaojin and Huang, Mengqi and Ding, Fei and He, Qian},
  journal={arXiv preprint arXiv:2509.06818},
  year={2025}
}
```
""".strip()

star = f"""
If UMO is helpful, please help to ⭐ our <a href='https://github.com/bytedance/UMO' target='_blank'> Github Repo</a> or cite our paper. Thanks a lot!
{article}
"""


def main(args):

    # Gradio
    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(badges_text)
        gr.Markdown(tips)

        with gr.Row():
            with gr.Column():
                # text prompt
                instruction = gr.Textbox(
                    label='Enter your prompt',
                    info='Use "first/second image" or “第一张图/第二张图” as reference.',
                    placeholder="Type your prompt here...",
                )

                with gr.Row(equal_height=True):
                    # input images
                    image_input_1 = gr.Image(label="First Image", type="pil")
                    image_input_2 = gr.Image(label="Second Image", type="pil")
                    image_input_3 = gr.Image(label="Third Image", type="pil")

                generate_button = gr.Button("Generate Image")

                negative_prompt = gr.Textbox(
                    label="Enter your negative prompt",
                    placeholder="Type your negative prompt here...",
                    value=NEGATIVE_PROMPT,
                )

                # slider
                with gr.Row(equal_height=True):
                    height_input = gr.Slider(
                        label="Height", minimum=256, maximum=2048, value=1024, step=128
                    )
                    width_input = gr.Slider(
                        label="Width", minimum=256, maximum=2048, value=1024, step=128
                    )
                
                with gr.Accordion("Speed Up Options", open=True):
                    with gr.Row(equal_height=True):
                        global enable_taylorseer
                        global enable_teacache
                        enable_taylorseer = gr.Checkbox(label="Using TaylorSeer to speed up", value=True)
                        enable_teacache = gr.Checkbox(label="Using TeaCache to speed up", value=False)
                    
                    with gr.Row(equal_height=True):
                        scheduler_input = gr.Dropdown(
                            label="Scheduler",
                            choices=["euler", "dpmsolver++"],
                            value="euler",
                            info="The scheduler to use for the model.",
                        )

                        num_inference_steps = gr.Slider(
                            label="Inference Steps", minimum=20, maximum=100, value=50, step=1
                        )
                
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row(equal_height=True):
                        align_res = gr.Checkbox(
                            label="Align Resolution",
                            info="Align output's resolution with the first reference image. Only valid when there is only one reference image.",
                            value=True
                        )
                    with gr.Row(equal_height=True):
                        text_guidance_scale_input = gr.Slider(
                            label="Text Guidance Scale",
                            minimum=1.0,
                            maximum=8.0,
                            value=5.0,
                            step=0.1,
                        )

                        image_guidance_scale_input = gr.Slider(
                            label="Image Guidance Scale",
                            minimum=1.0,
                            maximum=3.0,
                            value=2.0,
                            step=0.1,
                        )
                    with gr.Row(equal_height=True):
                        cfg_range_start = gr.Slider(
                            label="CFG Range Start",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                        )

                        cfg_range_end = gr.Slider(
                            label="CFG Range End",
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.1,
                        )
                    
                    def adjust_end_slider(start_val, end_val):
                        return max(start_val, end_val)

                    def adjust_start_slider(end_val, start_val):
                        return min(end_val, start_val)
                    
                    cfg_range_start.input(
                        fn=adjust_end_slider,
                        inputs=[cfg_range_start, cfg_range_end],
                        outputs=[cfg_range_end]
                    )

                    cfg_range_end.input(
                        fn=adjust_start_slider,
                        inputs=[cfg_range_end, cfg_range_start],
                        outputs=[cfg_range_start]
                    )

                    with gr.Row(equal_height=True):
                        num_images_per_prompt = gr.Slider(
                            label="Number of images per prompt",
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                        )

                        seed_input = gr.Slider(
                            label="Seed", minimum=-1, maximum=2147483647, value=-1, step=1
                        )
                    with gr.Row(equal_height=True):
                        max_input_image_side_length = gr.Slider(
                            label="max_input_image_side_length",
                            minimum=256,
                            maximum=2048,
                            value=2048,
                            step=256,
                        )
                        max_pixels = gr.Slider(
                            label="max_pixels",
                            minimum=256 * 256,
                            maximum=1536 * 1536,
                            value=1024 * 1024,
                            step=256 * 256,
                        )

            with gr.Column():
                with gr.Column():
                    # output image
                    output_image = gr.Image(label="Output Image")
                    global save_images
                    # save_images = gr.Checkbox(label="Save generated images", value=True)
                    save_images = True
                    with gr.Accordion("Examples Comparison with OmniGen2", open=False):
                        output_image_omnigen2 = gr.Image(label="Generated Image (OmniGen2)")
                    download_btn = gr.File(label="Download full-resolution", type="filepath", interactive=False)

        gr.Markdown(star)
        
        global accelerator
        global pipeline

        bf16 = True
        accelerator = Accelerator(mixed_precision="bf16" if bf16 else "no")
        weight_dtype = torch.bfloat16 if bf16 else torch.float32

        pipeline = load_pipeline(accelerator, weight_dtype, args)

        # click
        generate_button.click(
            run,
            inputs=[
                instruction,
                width_input,
                height_input,
                image_input_1,
                image_input_2,
                image_input_3,
                scheduler_input,
                num_inference_steps,
                negative_prompt,
                text_guidance_scale_input,
                image_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels,
                seed_input,
                align_res,
            ],
            outputs=[output_image, download_btn],
        )

        gr.Examples(
            examples=get_examples("assets/examples/OmniGen2"),
            inputs=[
                instruction,
                width_input,
                height_input,
                image_input_1,
                image_input_2,
                image_input_3,
                seed_input,
                align_res,
                output_image,
                output_image_omnigen2,
            ],
            label="We provide examples for academic research. The vast majority of images used in this demo are either generated or from open-source datasets. If you have any concerns, please contact us, and we will promptly remove any inappropriate content.",
            examples_per_page=15
        )

    # launch
    demo.launch(share=args.share, server_port=args.port, server_name=args.server_name, ssr_mode=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Share the Gradio app")
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to use for the Gradio app"
    )
    parser.add_argument(
        "--server_name", type=str, default=None
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="OmniGen2/OmniGen2",
        help="Path or HuggingFace name of the model to load."
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
        "--lora_path",
        type=str,
        default=None,
        help="Path to the LoRA checkpoint to load."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
