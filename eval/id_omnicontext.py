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
sys.path.append(os.path.abspath("projects/XVerse"))

import argparse
import datasets
import glob
import json

from PIL import Image, ImageOps
from tqdm import tqdm

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms

from projects.XVerse.eval.tools.face_id import FaceID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    return args

class Collator:
    def __call__(self, features):
        return features


def main(args: argparse.Namespace) -> None:
    accelerator = Accelerator(mixed_precision="bf16")

    face_score_model = FaceID(accelerator.device)
    pil2tensor = transforms.ToTensor()

    if os.path.exists(args.test_data):
        test_dataset = datasets.load_from_disk(args.test_data)
    else:
        test_dataset = datasets.load_dataset(args.test_data, split="train")
    loader = DataLoader(
        test_dataset,
        collate_fn=Collator(),
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    loader = accelerator.prepare(loader)

    matching_score, mismatching_score = {}, {}
    for i, batched_data in tqdm(enumerate(loader), total=len(loader), disable=accelerator.process_index!=0):
        data = batched_data[0]

        key = data["key"]
        task_type = data["task_type"]
        if task_type in ["single_object", "multi_object", "scene_object"]:
            continue
        if task_type not in matching_score:
            matching_score[task_type] = []
        if task_type not in mismatching_score:
            mismatching_score[task_type] = []
        input_images = data['input_images']
        input_images = [ImageOps.exif_transpose(img) for img in input_images]

        output_image_path = os.path.join(args.result_dir, args.model_name, "fullset", task_type, f"{key}.png")
        output_image = Image.open(output_image_path).convert("RGB")

        with torch.no_grad():
            ref_imgs = [(pil2tensor(img).unsqueeze(0)*255).to(torch.uint8) for img in input_images]
            tgt_img = (pil2tensor(output_image).unsqueeze(0)*255).to(torch.uint8)

            tgt_bboxes = face_score_model.detect(tgt_img)
            tgt_faces = [output_image.crop(bbox) for bbox in tgt_bboxes]

            ref_faces = []
            for i, ref_img in enumerate(ref_imgs):
                ref_bboxes_i = face_score_model.detect(ref_img)
                max_area, max_area_bbox = 0, None
                for ref_bbox_i in ref_bboxes_i:
                    area = (ref_bbox_i[2] - ref_bbox_i[0]) * (ref_bbox_i[3] - ref_bbox_i[1])
                    if area > max_area:
                        max_area, max_area_bbox = area, ref_bbox_i
                if max_area > 0:
                    ref_faces.append(input_images[i].crop(max_area_bbox))
            
            if len(ref_faces) > 0:
                for ref_face in ref_faces:
                    if len(tgt_faces) > 0:
                        similarity_scores = [face_score_model(ref_face, x) for x in tgt_faces]

                        similarity_scores = sorted(similarity_scores, reverse=True)
                        matching_score[task_type].append(similarity_scores[0] / 10.)
                        if len(similarity_scores) > 1:
                            _score = min(1, (similarity_scores[0] - similarity_scores[1]) / similarity_scores[0]) if similarity_scores[0] > 0 else 0
                            mismatching_score[task_type].append(_score * 10.)
                    else:
                        matching_score[task_type].append(0)
    
    name = "ID"
    temp_json = os.path.join(args.result_dir, args.model_name, name, f"_score_{accelerator.process_index}.json")
    os.makedirs(os.path.dirname(temp_json), exist_ok=True)
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump({
            "ID-Sim": matching_score,
            "ID-Conf": mismatching_score,
        }, f, indent=4, ensure_ascii=False)
    
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        all_matching_score, all_mismatching_score = {}, {}
        all_matching_score["avg"] = []
        all_mismatching_score["avg"] = []
        all_jsons = glob.glob(os.path.join(args.result_dir, args.model_name, name, f"_score_*.json"))
        for json_path in all_jsons:
            with open(json_path, "r") as f:
                rank_data = json.load(f)
            rank_matching_score, rank_mismatching_score = rank_data["ID-Sim"], rank_data["ID-Conf"]
            
            for task_type in rank_matching_score:
                if task_type not in all_matching_score:
                    all_matching_score[task_type] = []
                all_matching_score[task_type].extend(rank_matching_score[task_type])
                all_matching_score["avg"].extend(rank_matching_score[task_type])
            for task_type in rank_mismatching_score:
                if task_type not in all_mismatching_score:
                    all_mismatching_score[task_type] = []
                all_mismatching_score[task_type].extend(rank_mismatching_score[task_type])
                all_mismatching_score["avg"].extend(rank_mismatching_score[task_type])
        
        for task_type in all_matching_score:
            if len(all_matching_score[task_type]) > 0:
                all_matching_score[task_type] = sum(all_matching_score[task_type]) / len(all_matching_score[task_type])
        for task_type in all_mismatching_score:
            if len(all_mismatching_score[task_type]) > 0:
                all_mismatching_score[task_type] = sum(all_mismatching_score[task_type]) / len(all_mismatching_score[task_type])
        
        save_path = os.path.join(args.result_dir, args.model_name, name, "score.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "ID-Sim": all_matching_score,
                "ID-Conf": all_mismatching_score,
            }, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)