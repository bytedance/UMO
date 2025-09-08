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
BASE_DIR="projects/XVerse"
sys.path.append(os.path.abspath(BASE_DIR))

import argparse
import math
import numpy as np
import random
import shutil
import time

from collections import defaultdict
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch

from projects.XVerse.src.utils.data_utils import get_rank_and_worldsize, json_dump, json_load, split_grid, pil2tensor
from projects.XVerse.eval.tools.face_id import FaceID


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../examples")
    parser.add_argument("--test_list_name", type=str, default="base_test_list_200")
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    print(args)
    
    local_rank, global_rank, world_size = get_rank_and_worldsize()
    print(f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}")
    is_local_main_process = local_rank == 0
    is_main_process = global_rank == 0
    torch.cuda.set_device(local_rank)
    
    dtype = torch.bfloat16
    device = "cuda"
    
    run_name = time.strftime("%m%d_$H")
    face_score_model = FaceID(device)
    
    test_list = json_load(f"{BASE_DIR}/eval/tools/{args.test_list_name}.json", 'utf-8')
    images = list(glob(f"{args.input_dir}/*.png"))
    print(len(test_list), len(images))
    assert len(test_list) == len(images)
    
    num_samples = len(test_list)
    num_ranks = world_size
    assert local_rank == global_rank
    if world_size > 1:
        num_per_rank = math.ceil(num_samples / num_ranks)
        test_list_indices = list(range(num_samples))
        random.seed(0)
        random.shuffle(test_list_indices)
        local_test_list_indices = test_list_indices[local_rank*num_per_rank:(local_rank+1)*num_per_rank]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank % 8)
        print(f"[worker {local_rank}] got {len(local_test_list_indices)} local samples")

    run_name = time.strftime("%Y%m%d-%H")
    temp_dir = os.path.join(args.input_dir, f"eval_id_conf_temp_{run_name}")

    if is_main_process:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

    rank_json = {}
    with torch.no_grad():
        for i in tqdm(local_test_list_indices):
            test_sample = test_list[i]
            real_paths, real_faces, real_names = [], [], []
            for x in test_sample["modulation"][0]["src_inputs"]:
                img_path = x["image_path"]
                img_path = os.path.join(BASE_DIR, img_path)
                name = "_".join(img_path.split("/")[-2:])
                
                if name.startswith("human"):
                    real_paths.append(img_path)
                    try:
                        real_faces.append(Image.open(img_path).convert("RGB"))
                        real_names.append(name)
                    except Exception as e:
                        print(f"Failed to open image {img_path}, error message: {e}")

            gen_img_path = list(filter(lambda x: x.split("/")[-1].split("_")[0].split(".")[0] == str(i), images))[0]
            rank_json[i] = []
            try:
                for j, gen_img in enumerate(split_grid(Image.open(gen_img_path))):
                    rank_json[i].append({})
                    if len(real_names) > 0:
                        gen_bboxes = face_score_model.detect(
                            (pil2tensor(gen_img).unsqueeze(0) * 255).to(torch.uint8)
                        )
                        gen_faces = [gen_img.crop(bbox) for bbox in gen_bboxes]
                        for k, (real_name, real_face) in enumerate(zip(real_names, real_faces)):
                            if len(gen_faces) > 1:
                                similarity_scores = [face_score_model(real_face, x) for x in gen_faces]
                                similarity_scores_top2 = sorted(similarity_scores, reverse=True)[:2]
                                score = min(1, (similarity_scores_top2[0] - similarity_scores_top2[1]) / similarity_scores_top2[0]) if similarity_scores_top2[0] > 0 else 0
                                rank_json[i][j][real_name] = score
            except Exception as e:
                print(f"Failed to process image {gen_img_path}, error message: {e}")
                
    json_dump(rank_json, f"{temp_dir}/scores_{global_rank}.json", "utf-8")

    if is_main_process:
        while len(glob(f"{temp_dir}/scores_*.json")) < world_size:
            time.sleep(5)
        time.sleep(5) # wait for the file writting to be finished
        merged_json = {}
        id_scores = defaultdict(list)
        all_scores = []
        for rank_path in glob(f"{temp_dir}/scores_*.json"):
            rank_json = json_load(rank_path, "utf-8")
            merged_json.update(rank_json)

        for i, grid_json in merged_json.items():
            for img_json in grid_json:
                for id_name, id_score in img_json.items():
                    id_scores[id_name].append(id_score*100)
        
        for id_name in id_scores:
            all_scores += id_scores[id_name]
            id_scores[id_name] = np.mean(id_scores[id_name])
            print(id_name, id_scores[id_name])

        prefix = "id_conf"
        json_dump(merged_json, f"{args.input_dir}/{prefix}_{run_name}.json", "utf-8")
        final_id_score = np.mean(all_scores)
        lines_to_write = [
            f"ID Conf: {final_id_score:.2f}\n"
        ]
        print(lines_to_write[0])
        for id_name, score in id_scores.items():
            lines_to_write.append(f"{id_name}: {score:.2f}\n")

        with open(f"{args.input_dir}/{prefix}_{run_name}.txt", "w") as f:
            f.writelines(lines_to_write)

        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()