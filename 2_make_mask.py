import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

# dataset_list = ['statue', 'dinosaur', 'shoe', 'bear']
dataset_list = ['dinosaur', 'bear']



root_dir = os.path.join(os.getcwd(), 'data', 'nerf_llff_data')

# 
for dataset in tqdm(dataset_list):
    # read mask imgs in the dataset
    data_root_dir = os.path.join(root_dir, dataset)
    # img path : prompts/dataset/img
    img_dir_path = os.path.join(data_root_dir, 'prompts', dataset, 'img')
    # print(img_dir_path)

    for img in os.listdir(img_dir_path):
        # print(cv2.imread(os.path.join(img_dir_path, img)).shape)
        img_path = os.path.join(img_dir_path, img)
        mask_path = os.path.join(data_root_dir, 'prompts', dataset, 'mask', img)
        i = cv2.imread(img_path)
        resized_img = cv2.resize(i, (64, 64))
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        # print numpy array unique values
        # print(np.unique(gray_img)) -> [0 128 255]
        _, binary_mask = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        tensor_mask = torch.from_numpy(binary_mask).unsqueeze(0)
        # print(tensor_mask.shape) -> torch.Size([1, 64, 64])

        # Save the tensor to a .pt file
        tensor_path = os.path.splitext(mask_path)[0] + '.pt'
        # print(tensor_path)
        torch.save(tensor_mask, tensor_path)
        