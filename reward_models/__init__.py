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

def load_reward_models(args, device, weight_dtype):
    reward_models, reward_image_transforms, reward_lambdas = {}, {}, {}
    
    if "MIMR" in args.reward_model_id:
        from reward_models.MIMR.multi_id_matching_reward import MIMR
        reward_models["MIMR"] = MIMR(device, weight_dtype).train(False).requires_grad_(False)
        reward_image_transforms["MIMR"] = lambda x: x # [c, h, w] -1~1
    
    for reward_model_idx, reward_model_id in enumerate(args.reward_model_id):
        reward_lambdas[reward_model_id] = args.reward_lambda[reward_model_idx]
    return reward_models, reward_image_transforms, reward_lambdas