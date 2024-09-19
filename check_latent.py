import os
import torch
import cv2

custom_path = "/home/cvlab/Desktop/kuaicv/prev_Tensorf_latent_iclr/data/nerf_llff_data/face/prompts/face/mask/frame_00001.th"
origin_path = "./data/nerf_llff_data/fortress/latents/001_l.pt"

c_l = torch.load(custom_path)
o_l = torch.load(origin_path)

print(c_l.shape)
print(o_l.shape)

custom_path = "./data/nerf_llff_data/bear/images_l_64/frame_00001.jpg"
origin_path = "./data/nerf_llff_data/fortress/images_l_64/001.png"


c_i = cv2.imread(custom_path)
o_i = cv2.imread(origin_path)

print(c_i.shape)
print(o_i.shape)
