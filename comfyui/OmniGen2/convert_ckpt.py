import argparse

from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="models/bytedance-research/UMO/UMO_OmniGen2.safetensors")
parser.add_argument("--output_path", type=str, default="models/bytedance-research/UMO/UMO_OmniGen2_comfyui.safetensors")
args = parser.parse_args()

lora_ckpt = load_file(args.ckpt_path, device="cpu")
new_ckpt = {}
for k in lora_ckpt.keys():
    new_k = "diffusion_model." + k.replace(".default.", ".")
    new_ckpt[new_k] = lora_ckpt[k]

save_file(new_ckpt, args.output_path)