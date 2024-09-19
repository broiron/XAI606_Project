import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
from diffusers import AutoencoderKL

_rearrange2 = lambda x: (einops.rearrange(x, '(b h w c) -> b c h w', b=1, h=64, w=64))
_rearrange = lambda x: (einops.rearrange(x, '(b h w) c -> b c h w', b=1, c=4, h=64))
def OctreeRender_trilinear_deform(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda', deform_mlp=None):

    latents, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        # rgb_map, depth_map = tensorf.forward(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
        feature_map, depth_map, weights, alphas, acc_map, deform_matching_loss = tensorf.forward_deform(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples, deform_mlp=deform_mlp)

        latents.append(feature_map)
        depth_maps.append(depth_map)
    
    return torch.cat(latents), alphas, torch.cat(depth_maps), weights, acc_map, deform_matching_loss

def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    latents, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        # rgb_map, depth_map = tensorf.forward(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
        feature_map, depth_map, weights, alphas, acc_map = tensorf.forward_edit(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        latents.append(feature_map)
        depth_maps.append(depth_map)
    
    return torch.cat(latents), alphas, torch.cat(depth_maps), weights, acc_map

def OctreeRender_trilinear_fast_depth(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    latents, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        # rgb_map, depth_map = tensorf.forward(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
        feature_map, depth_map, weights, alphas, acc_map = tensorf.forward_edit_depth(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        latents.append(feature_map)
        depth_maps.append(depth_map.detach())
    
    return torch.cat(latents), alphas, torch.cat(depth_maps), weights, acc_map

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # To the GPU we go!
    vae = vae.to(device)
    vae_magic = 0.18215 # vae model trained with a scale term to get closer to unit variance
    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, weight_map, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.reshape(H,W,4)
        rgb_map = rgb_map.permute(2, 0, 1).unsqueeze(0)
        rgb_map = latents2images(rgb_map, vae, vae_magic)
        rgb_map = rgb_map[0]
        depth_map = depth_map.reshape(H,W,1).cpu()
        
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        depth_map = Image.fromarray(depth_map)
        depth_map = depth_map.resize((512,512))

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}_d.png', depth_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    return PSNRs


@torch.no_grad()
def evaluation_deform(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda', deform_mlp=None):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # To the GPU we go!
    vae = vae.to(device)
    vae_magic = 0.18215 # vae model trained with a scale term to get closer to unit variance
    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, weight_map, _, = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device, deform_mlp=deform_mlp)
        rgb_map = rgb_map.reshape(H,W,4)
        rgb_map = rgb_map.permute(2, 0, 1).unsqueeze(0)
        rgb_map = latents2images(rgb_map, vae, vae_magic)
        depth_map = depth_map.reshape(H,W,1).cpu()
        
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        depth_map = Image.fromarray(depth_map)
        depth_map = depth_map.resize((512,512))

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}_d.png', depth_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    return PSNRs

@torch.no_grad()
def evaluation_vae(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda', small_vae=None):
    PSNRs, rgb_maps, depth_maps, vae_outputs = [], [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    os.makedirs(savePath+"/raw", exist_ok=True)

    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # To the GPU we go!
    vae = vae.to(device)
    vae_magic = 0.18215 # vae model trained with a scale term to get closer to unit variance

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far

    # print(test_dataset.all_rays)

    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))

    # print("ray_shape: ", test_dataset.all_rays.shape[0])

    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        # print(rays.shape) # 

        raw_map, _, depth_map, weight_map, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        
        # print("raw_map shape: ", raw_map.shape)

        residual = _rearrange(raw_map)
        depth_map = _rearrange2(depth_map) 
        vae_output = small_vae.forward(residual)
        vae_output, vae_output_npy = latents2images(vae_output, vae, vae_magic)

        raw_map = _rearrange(raw_map)
        raw_map, raw_map_npy = latents2images(raw_map, vae, vae_magic)
        depth_map = depth_map.reshape(H,W,1).cpu()
        
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        depth_map = Image.fromarray(depth_map)
        depth_map = depth_map.resize((512,512))

        if len(test_dataset.all_rgbs):
            gt_latent = test_dataset.all_features[idxs[idx]]
            _, gt_rgb_npy = latents2images(gt_latent, vae, vae_magic)
            loss = torch.mean(torch.tensor(vae_output_npy - gt_rgb_npy) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

        vae_outputs.append(vae_output)
        rgb_maps.append(raw_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/raw/{prtx}{idx:03d}.png', raw_map)
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}_vae.png', vae_output)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}_d.png', depth_map)

    imageio.mimwrite(f'{savePath}/{prtx}vae_video.mp4', np.stack(vae_outputs), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs

@torch.no_grad()
def evaluation_sampling(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # To the GPU we go!
    vae = vae.to(device)
    vae_magic = 0.18215
    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, weight_map, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)

        mean, logvar = torch.chunk(rgb_map, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)
        eps = torch.randn(mean.shape).to(device=mean.device)
        rgb_map = mean + std * eps

        rgb_map = rgb_map.reshape(H,W,4)
        rgb_map = rgb_map.permute(2, 0, 1).unsqueeze(0)
        rgb_map = latents2images(rgb_map, vae, vae_magic)
        depth_map = depth_map.reshape(H,W,1).cpu()
        
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        depth_map = Image.fromarray(depth_map)
        depth_map = depth_map.resize((512,512))

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}_d.png', depth_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    os.makedirs(savePath+"/latents", exist_ok=True)
    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # To the GPU we go!
    vae = vae.to(device)
    vae_magic = 0.18215 # vae model trained with a scale term to get closer to unit variance

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):
        W, H = test_dataset.img_wh
        # linear_decoding = linear_decoding.expand(-1, -1, -1, H, W)
        # linear_decoding = linear_decoding.to(device)
        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.reshape(H,W,4)
        rgb_map = rgb_map.permute(2, 0, 1).unsqueeze(0)

        torch.save(rgb_map, f'{savePath}/latents/{idx:03d}.th')
        rgb_map = latents2images(rgb_map, vae, vae_magic)   # [1,4,64,64] latent needed
        depth_map = depth_map.reshape(H,W,1).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        depth_map = Image.fromarray(depth_map)
        depth_map = depth_map.resize((512,512))
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}_d.png', depth_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)


    return PSNRs

def evaluation_latents(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/latents", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):
        W, H = test_dataset.img_wh
        # linear_decoding = linear_decoding.expand(-1, -1, -1, H, W)
        # linear_decoding = linear_decoding.to(device)
        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        # if ndc_ray: # latent Nerf에서 llff인데 no_ndc일때, ndc_rays_blender 써야함.
        # rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.reshape(H,W,4)
        rgb_map = rgb_map.permute(2, 0, 1).unsqueeze(0)

        torch.save(rgb_map, f'{savePath}/latents/{idx:03d}.th')

    return 0

def latents2images(latents, vae, vae_magic):
    latents = latents/vae_magic
    latents = latents.to(vae.device)

    vae = vae.to('cuda:0')
    latents = latents.to('cuda:0')
    with torch.no_grad():
        imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0,1)
    imgs = imgs.detach().cpu().permute(0,2,3,1).numpy() # (1,512,512,3)
    imgs = (imgs * 255).round().astype("uint8")
    imgs = [Image.fromarray(i) for i in imgs]
    # when imgs are not multiple images, using this
    imgs = imgs[0]
    return imgs 

def latents2images(latents, vae, vae_magic):
    latents = latents/vae_magic
    latents = latents.to(vae.device)

    # vae = vae.to(latents.device)
    # latents = latents.to('cuda:0')
    with torch.no_grad():
        imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0,1)
    imgs_npy = imgs.detach().cpu().permute(0,2,3,1).numpy() # (1,512,512,3)
    imgs = (imgs_npy * 255).round().astype("uint8")
    imgs = [Image.fromarray(i) for i in imgs]
    # when imgs are not multiple images, using this
    imgs = imgs[0]
    return imgs, imgs_npy

def latents2images_taesd(latents, TAESD):
    latents = latents
    with torch.no_grad():
        imgs = TAESD.decoder(latents)
    imgs = imgs.clamp(0,1)
    imgs = imgs.detach().cpu().permute(0,2,3,1).numpy()
    imgs = (imgs * 255).round().astype("uint8")
    imgs = [Image.fromarray(i) for i in imgs]
    return imgs[0]
    