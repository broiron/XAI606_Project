import os
import numpy as np

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

from PIL import Image
import torch, logging
# from torch import autocast
from torch.cuda.amp import autocast
from torchvision import transforms as tfms

from huggingface_hub import notebook_login
from diffusers import AutoencoderKL,UNet2DConditionModel,LMSDiscreteScheduler,StableDiffusionInpaintPipeline
import glob
import cv2

# Set device
torch_device = "cuda:1" if torch.cuda.is_available() else "cpu"

# summarize tensor
_s = lambda x: (x.shape,x.max(),x.min())

# Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# To the GPU we go!
vae = vae.to(torch_device)
vae_magic = 0.18215 # vae model trained with a scale term to get closer to unit variance

def image2latent(im):
    im = tfms.ToTensor()(im).unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(im.to(device = torch_device)*2-1);
    latent = latent.latent_dist.sample() * vae_magic      
    return latent

def img_to_latent_and_rgb(im_path):
    im_path = Path(im_path)
    im = Image.open(im_path).resize((512,512))
    im_latent = image2latent(im)
    v1_4_rgb_latent_factors_v1 = [
    #   R       G       B
    [ 0.298,  0.207,  0.208],  # L1
    [ 0.187,  0.286,  0.173],  # L2
    [-0.158,  0.189,  0.264],  # L3
    [-0.184, -0.271, -0.473],  # L4
    ]

    v1_4_rgb_latent_factors_v1 = torch.tensor(v1_4_rgb_latent_factors_v1) #.to(device=torch_device)

    im_latent = im_latent.detach().cpu()

    v1_4_rgb_latent_factors_v1 = v1_4_rgb_latent_factors_v1.view(1, 4, 3)
    v1_4_rgb_latent_factors_v1 = v1_4_rgb_latent_factors_v1.unsqueeze(-1).unsqueeze(-1)
    v1_4_rgb_latent_factors_v1 = v1_4_rgb_latent_factors_v1.expand(-1, -1, -1, 64, 64)


    # perform element-wise multiplication and summation along the second dimension
    result1 = torch.einsum('n c h w, n c r h w -> n r h w', im_latent, v1_4_rgb_latent_factors_v1)

    return result1
#root_dir = './data/nerf_llff_data/fortress/images_4'
#load image path
#image_paths = sorted(glob.glob(os.path.join(root_dir, 'images_4/*')))


#decode_direct1, im_latent = img_to_latent_and_rgb(image_paths)
# Normalizing latent image to 0 ~ 1
#decode_direct1 = (decode_direct1-decode_direct1.min())/(decode_direct1.max()-decode_direct1.min())