#!/bin/bash

set -e


export FACE_ID_MODEL_PATH="projects/XVerse/checkpoints/model_ir_se50.pth"

pip install webdataset==0.2.111

model_name=$1

accelerate launch \
    -m eval.id_omnicontext \
        --test_data OmniGen2/OmniContext \
        --result_dir output/OmniContext \
        --model_name $model_name