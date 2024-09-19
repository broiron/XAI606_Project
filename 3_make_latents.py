import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from diffusers import AutoencoderKL
from torchvision.transforms.transforms import *
from torchvision import transforms as tfms
from PIL import Image
from dataLoader.img2latent import *


# dataset_list = ['statue', 'dinosaur', 'shoe', 'bear']
dataset_list = ['bear', 'dinosaur']


root_dir = os.path.join(os.getcwd(), 'data', 'nerf_llff_data')

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# To the GPU we go!
vae = vae.to(torch_device)
vae_magic = 0.18215 # vae model trained with a scale term to get closer to unit variance
vae.requires_grad_(False)

def image2latent_mine(im):
    im = tfms.ToTensor()(im).unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(im.to(device = torch_device)*2-1);
    latent = latent.latent_dist.sample() * vae_magic      
    return latent


for dataset in tqdm(dataset_list):
    # read mask imgs in the dataset
    data_root_dir = os.path.join(root_dir, dataset)
    # img path : prompts/dataset/img
    img_dir_path = os.path.join(data_root_dir, 'images')
    # print(img_dir_path)
    latent_dir_path = os.path.join(data_root_dir, 'latents')
    
    if not os.path.exists(latent_dir_path):
        os.makedirs(latent_dir_path)
    
    l_64_dir_path = os.path.join(data_root_dir, 'images_l_64')
    r_64_dir_path = os.path.join(data_root_dir, 'images_r_64')

    if not os.path.exists(r_64_dir_path):
        os.makedirs(r_64_dir_path)
    if not os.path.exists(l_64_dir_path):
        os.makedirs(l_64_dir_path)


    if not os.path.exists(l_64_dir_path):
        os.makedirs(l_64_dir_path)

    for img in os.listdir(img_dir_path):
        # print(cv2.imread(os.path.join(img_dir_path, img)).shape)
        img_path = os.path.join(img_dir_path, img)
        im = Image.open(img_path).resize((512,512))
        im_64 = im.resize((64,64))

        # save 64x64 image to l_64_dir_path
        im_64.save(os.path.join(r_64_dir_path, img))

        im_latent = image2latent_mine(im) # [1, 4, 64, 64]

        latent_image = img_to_latent_and_rgb(im_path=img_path) # [1, 3, 64, 64]
        
        # transfrom to numpy image to save
        latent_image = latent_image.squeeze(0).permute(1,2,0).numpy()

        # save latent image to l_64_dir_path
        latent_image = (latent_image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(l_64_dir_path, img), latent_image)


        # save latent to .pt file
        latent_path = os.path.splitext(img)[0] + '_l.pt'
        # img+ '_l.pt'
        latent_path = os.path.join(latent_dir_path, latent_path)
        torch.save(im_latent, latent_path)
        
        
        