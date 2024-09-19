import os
from tqdm import tqdm
# dataset_list = ['statue', 'dinosaur', 'shoe', 'bear']
dataset_list = ['bear', 'dinosaur']

root_dir = os.path.join(os.getcwd(), 'data', 'nerf_llff_data')

# 
for dataset in dataset_list:
    # make prompts directory
    os.makedirs(os.path.join(root_dir, dataset, 'prompts'), exist_ok=True)
    # move images dir to prompts dir
    # os.rename(os.path.join(root_dir, dataset, dataset), os.path.join(root_dir, dataset, 'prompts', dataset))
    
    os.makedirs(os.path.join(root_dir, dataset, 'prompts', dataset, 'img'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, dataset, 'prompts', dataset, 'mask'), exist_ok=True)

    # move images in os.path.join(root_dir, dataset, 'prompts', dataset) to img dir
    mask_img_dir = os.path.join(root_dir, dataset, 'prompts', dataset, 'img')
    for filename in tqdm(os.listdir(os.path.join(root_dir, dataset, dataset))):
        if filename.endswith('.png'):
            new_filename = str(int(filename[6:-4].lstrip('0'))-1).zfill(3) + '.png'
            # new_filename = filename
            os.rename(os.path.join(root_dir, dataset, dataset, filename), os.path.join(mask_img_dir, new_filename))
    
    img_dir = os.path.join(root_dir, dataset, 'images')

    for filename in tqdm(os.listdir(img_dir)):
        if filename.endswith('.jpg'):
            
            new_filename = str(int(filename[6:-4].lstrip('0'))-1).zfill(3) + '.jpg'
            # new filename starts from 000.jpg
            os.rename(os.path.join(root_dir, dataset, 'images', filename), os.path.join(root_dir, dataset, 'images', new_filename))
    
