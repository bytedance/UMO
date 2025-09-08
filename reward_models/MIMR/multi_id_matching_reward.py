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

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_affine

import numpy as np
from facexlib.detection import init_detection_model
from scipy.optimize import linear_sum_assignment
from skimage import transform as trans

from .arcface_arch import get_arcface_model

class MIMR(nn.Module):
    def __init__(self, device, weight_dtype):
        super().__init__()
        self.device = device
        self.weight_dtype = weight_dtype

        # detect fact to get bbox
        self.det_net = init_detection_model('retinaface_resnet50', half=False, device=device, model_rootpath='models/pulid/facexlib').requires_grad_(False).to(device, dtype=weight_dtype)
        self.det_net.mean_tensor = self.det_net.mean_tensor.to(dtype=weight_dtype)

        self.template_landmark = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        self.template_landmark[:, 0] += 8.0
        self.tform = trans.SimilarityTransform()

        # get face emb
        self.arcface_model = get_arcface_model(model_name='r50', pretrained_path='models/pulid/glint_cosface_res50.pth').to(device, dtype=weight_dtype)
    
    @torch.no_grad()
    def get_bbox(self, image):
        bs = 1
        image = image * 0.5 + 0.5
        # prepare data following facexlib
        reverse_index = torch.tensor([2, 1, 0], device=self.device)
        mean_tensor = torch.tensor([[[[104.]], [[117.]], [[123.]]]], device=self.device, dtype=self.weight_dtype)
        bgr_image = image[:, reverse_index]
        bgr_image = bgr_image * 255. - mean_tensor
        bgr_image = bgr_image.permute(0, 2, 3, 1)
        bboxs, landmarks = self.det_net.batched_detect_faces(bgr_image)
        # bboxs: list of np.array ([n_boxes, 5], type=np.float32); landmarks: list of np.array ([n_boxes, 10], type=np.float32)
        bboxs = bboxs[0] # bs=1; np.array ([n_boxes, 5], type=np.float32)
        landmarks = landmarks[0] # bs=1; np.array ([n_boxes, 10], type=np.float32)
        num_face = bboxs.shape[0]
        
        Ms = []
        if num_face == 0:
            Ms.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
            id_weight = torch.zeros((1,), device=self.device)
        else:
            id_weight = torch.ones((num_face,), device=self.device)
            for face_idx in range(num_face):
                landmark = np.array([[landmarks[face_idx][j], landmarks[face_idx][j + 1]] for j in range(0, 10, 2)])
                self.tform.estimate(landmark, self.template_landmark)
                M = self.tform.params[0:2, :]
                Ms.append(M.astype(np.float32))
        Ms = torch.from_numpy(np.stack(Ms, axis=0)).to(self.device) # (num_face, 2, 3)
        return Ms, id_weight

    def score_grad(self, image: torch.Tensor, image_gt: torch.Tensor, **kwargs):
        # image: [c, h, w] -1~1
        image = image.unsqueeze(0)
        image_gt = image_gt.unsqueeze(0)

        with torch.no_grad():
            M_gt, id_weight_gt = self.get_bbox(image_gt.to(self.device).to(self.weight_dtype))
            num_face_gt = M_gt.shape[0]
            face_gt = warp_affine(image_gt.repeat(num_face_gt, 1, 1, 1).to(torch.float32).to(self.device), M_gt, (112, 112)).to(self.weight_dtype) # (num_face, c, 112, 112), -1~1
            face_emb_gt = self.arcface_model(face_gt) # (num_face, 512)
            face_emb_norm_gt = F.normalize(face_emb_gt, dim=-1)
        
        M_pred, id_weight_pred = self.get_bbox(image.to(self.device).to(self.weight_dtype))
        num_face_pred = M_pred.shape[0]
        face_pred = warp_affine(image.repeat(num_face_pred, 1, 1, 1).to(torch.float32).to(self.device), M_pred, (112, 112)).to(self.weight_dtype)
        face_emb_pred = self.arcface_model(face_pred)
        face_emb_pred_gt = F.normalize(face_emb_pred, dim=-1)

        sim_matrix = torch.matmul(face_emb_norm_gt, face_emb_pred_gt.T) # (num_face_gt, num_face_pred)

        with torch.no_grad():
            row_ind, col_ind = linear_sum_assignment(-sim_matrix.to(torch.float32).cpu().numpy())
        
        weight_matrix = torch.ones_like(sim_matrix) * -1.
        for r, c in zip(row_ind, col_ind):
            weight_matrix[r, c] = 1
        
        id_weight_matrix = torch.matmul(id_weight_gt.unsqueeze(-1), id_weight_pred.unsqueeze(0))

        reward = (sim_matrix * weight_matrix * id_weight_matrix).mean()
        
        return reward