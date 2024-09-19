#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from loss_utils import ssim
import json
from tqdm import tqdm
from image_utils import psnr
from argparse import ArgumentParser

########################

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import natsort

class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text):
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image):  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(
        self, image_0, image_1, text_0, text_1
    ):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_0, sim_1, sim_direction, sim_image
    
########################

from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn

class Dinov2Similarity:
    def __init__(self, name: str = "facebook/dinov2-base"):
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to('cuda')
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(
        self, image_0, image_1 # receive PIL images
    ):
        with torch.no_grad():
            inputs1 = self.processor(images=image_0, return_tensors="pt").to('cuda')
            outputs1 = self.model(**inputs1)
            image_features1 = outputs1.last_hidden_state
            image_features1 = image_features1.mean(dim=1)

            inputs2 = self.processor(images=image_1, return_tensors="pt").to('cuda')
            outputs2 = self.model(**inputs2)
            image_features2 = outputs2.last_hidden_state
            image_features2 = image_features2.mean(dim=1)

        sim = self.cos(image_features1[0], image_features2[0]).item()
        sim = (sim+1) / 2
        return sim
        
########################

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []

    renders_pil = []
    gts_pil = []

    image_names = []

    
    renders_dir_list = os.listdir(renders_dir)
    # only take the files that ends with .png
    renders_dir_list = [x for x in renders_dir_list if x.endswith(".png")]

    for fname in natsort.natsorted(renders_dir_list):
        render = Image.open(renders_dir / fname)
        renders_pil.append(render)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)


    # print(image_names)

    for fname in natsort.natsorted(os.listdir(gt_dir)):
        gt = Image.open(gt_dir / fname)
        gts_pil.append(gt)
        image_names.append(fname)
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
    
    # print(image_names)

    return renders, gts, image_names, renders_pil, gts_pil

def evaluate(model_paths, gt_paths, source_prompt, target_prompt):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    print("loading clip...")
    clip_similarity = ClipSimilarity("ViT-L/14").to(device)

    print("loading dino...")
    dinov2_similarity = Dinov2Similarity("facebook/dinov2-base")

    print("")
    
    scene_dir = model_paths[0]

    print("Scene:", scene_dir)
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "save" 

    renders_dir = Path(scene_dir) / "imgs_render_all"
    gt_dir = Path(gt_paths[0]) / "images"
    # method : ours_{iteration}

    full_dict[renders_dir] = {}
    per_view_dict[renders_dir] = {}
    full_dict_polytopeonly[renders_dir] = {}
    per_view_dict_polytopeonly[renders_dir] = {}


    renders, gts, image_names, renders_pil, gts_pil = readImages(renders_dir, gt_dir)

    # renders: editing된 image, gts: source image

    clip_direction_sim = []
    clip_img_sim = []
    dino_sims = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        
        # sim_direction, sim_image만 사용
        sim_0, sim_1, sim_direction, sim_image = clip_similarity.forward(gts[idx], renders[idx], source_prompt, target_prompt)
        dino_sim = dinov2_similarity.forward(renders_pil[idx], gts_pil[idx])

        clip_direction_sim.append(sim_direction)
        clip_img_sim.append(sim_image)
        dino_sims.append(dino_sim)

    print("  CLIP Text-Image Directional Score : {:>12.7f}".format(torch.tensor(clip_direction_sim).mean(), ".5"))
    print("  CLIP Image-Image Similarity Score : {:>12.7f}".format(torch.tensor(clip_img_sim).mean(), ".5"))
    print("  DINO Similarity Score : {:>12.7f}".format(torch.tensor(dino_sims).mean(), ".5"))
    print("")

    #full_dict[scene_dir][method].update({"CLIP Text-Image Directional Score": torch.tensor(clip_direction_sim).mean().item(),
    #                                        "CLIP Image-Image Similarity Score": torch.tensor(clip_img_sim).mean().item(),
    #                                        "DINO Similarity Score": torch.tensor(dino_sims).mean().item()})

    #per_view_dict[scene_dir][method].update({"CLIP Text-Image Directional Score": {name: clip_direction for clip_direction, name in zip(torch.tensor(clip_direction_sim).tolist(), image_names)},
    #                                        "CLIP Image-Image Similarity Score": {name: clip_img for clip_img, name in zip(torch.tensor(clip_img_sim).tolist(), image_names)},
    #                                        "DINO Similarity Score": {name: dino_sim for dino_sim, name in zip(torch.tensor(dino_sims).tolist(), image_names)}})

    #with open(scene_dir + "/results.json", 'w') as fp:
    #    json.dump(full_dict[scene_dir], fp, indent=True)
    #with open(scene_dir + "/per_view.json", 'w') as fp:
    #    json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--gt_paths', '-g', required=True, nargs="+", type=str, default=[])

    parser.add_argument('--source_prompt', '-sp', type=str, default="a photo of red flowers")
    parser.add_argument('--target_prompt', '-tp', type=str, default="a photo of cherry blossoms")


    args = parser.parse_args()
    evaluate(args.model_paths, args.gt_paths, args.source_prompt, args.target_prompt)
