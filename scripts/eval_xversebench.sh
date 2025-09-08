#!/bin/bash

set -e


export FLORENCE2_MODEL_PATH="projects/XVerse/checkpoints/Florence-2-large"
export SAM2_MODEL_PATH="projects/XVerse/checkpoints/sam2.1_hiera_large.pt"
export FACE_ID_MODEL_PATH="projects/XVerse/checkpoints/model_ir_se50.pth"
export CLIP_MODEL_PATH="projects/XVerse/checkpoints/clip-vit-large-patch14"
export FLUX_MODEL_PATH="projects/XVerse/checkpoints/FLUX.1-dev"
export DPG_VQA_MODEL_PATH="projects/XVerse/checkpoints/mplug_visual-question-answering_coco_large_en"
export DINO_MODEL_PATH="projects/XVerse/checkpoints/dino-vits16"


task_type="$1"
test_list_name="XVerseBench_$task_type"

dir_name="$2"
save_name=$(realpath "$dir_name")

if [[ "$test_list_name" == "XVerseBench_multi" ]]; then
    accelerate launch \
        -m eval.id_conf_xversebench \
        --input_dir "$save_name" \
        --test_list_name "$test_list_name"
fi

cd projects/XVerse

accelerate launch \
    -m eval.tools.idip_dpg_score \
    --input_dir "$save_name" \
    --test_list_name "$test_list_name"

accelerate launch \
    -m eval.tools.idip_aes_score \
    --input_dir "$save_name" \
    --test_list_name "$test_list_name"

accelerate launch \
    -m eval.tools.idip_face_score \
    --input_dir "$save_name" \
    --test_list_name "$test_list_name"

accelerate launch \
    -m eval.tools.idip_sam-dino_score \
    --input_dir "$save_name" \
    --test_list_name "$test_list_name"

python \
    -m eval.tools.log_scores \
    --input_dir "$save_name" \
    --test_list_name "$test_list_name"