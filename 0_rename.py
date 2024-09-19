import os

dataset_list = ['dinosaur', 'bear']

root_dir = os.path.join(os.getcwd(), 'data', 'nerf_llff_data')

for dataset in dataset_list:
    
    data_root_dir = os.path.join(root_dir, dataset)
    mask_img_dir_path = os.path.join(data_root_dir, 'prompts', dataset, 'img')

    __img_dir_path = os.path.join(data_root_dir, '__images')
    img_dir_path = os.path.join(data_root_dir, 'images')
    
    for filename in os.listdir(__img_dir_path):
        
        new_filename = filename[6:-4].lstrip('0').zfill(3) + '.jpg'

        os.rename(os.path.join(__img_dir_path, filename), os.path.join(img_dir_path, new_filename))

    for filename in os.listdir(mask_img_dir_path):
        
        new_filename = filename[6:-4].lstrip('0').zfill(3) + '.png'

        os.rename(os.path.join(mask_img_dir_path, filename), os.path.join(mask_img_dir_path, new_filename))