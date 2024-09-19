import cv2,torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.signal
import einops
from torchvision import transforms as tfms
import math

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


class CosineDecayScheduler:
    def __init__(self, max_iterations, base_lr, min_lr):
        self.max_iterations = max_iterations
        self.base_lr = base_lr
        self.min_lr = min_lr

    def get_lr(self, iteration):
        cos_decay = 0.5 * (1 + math.cos(math.pi * iteration / self.max_iterations))
        lr = self.min_lr + (self.base_lr - self.min_lr) * cos_decay
        return lr

class CosineAscendScheduler:
    def __init__(self, max_iterations, base_lr, max_lr):
        self.max_iterations = max_iterations
        self.base_lr = base_lr
        self.max_lr = max_lr

    def get_lr(self, iteration):
        cos_ascent = 0.5 * (1 - math.cos(math.pi * iteration / self.max_iterations))
        lr = self.base_lr + (self.max_lr - self.base_lr) * cos_ascent
        return lr

def latents2img(latents, vae):
    vae_magic = 0.18215
    latents = latents/vae_magic
    vae = vae.to(latents.device)
    with torch.no_grad():
        imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0,1)
    imgs = imgs.detach().cpu().permute(0,2,3,1).numpy()
    imgs = (imgs * 255).round().astype("uint8")
    imgs_np = imgs
    imgs = [Image.fromarray(i) for i in imgs]
    return imgs, imgs_np

def latents2img_w_grad(latents, vae):
    vae_magic = 0.18215
    latents = latents/vae_magic
    vae = vae.to(latents.device)
    imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0,1)
    imgs_tensor = imgs
    imgs = imgs.detach().cpu().permute(0,2,3,1).numpy()
    imgs = (imgs * 255).round().astype("uint8")
    imgs_np = imgs
    imgs = [Image.fromarray(i) for i in imgs]
    return imgs, imgs_tensor

def latents2img_taesd(latents, TAESD):
    latents = latents
    with torch.no_grad():
        imgs = TAESD.decoder(latents)
    imgs = imgs.clamp(0,1)
    imgs_tensor = imgs
    imgs = imgs.detach().cpu().permute(0,2,3,1).numpy()
    imgs = (imgs * 255).round().astype("uint8")
    imgs = [Image.fromarray(i) for i in imgs]
    return imgs[0], imgs_tensor


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0

    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x[x<1])
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    # x = 1-x   # For getting inverse depth map
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    # x_ = np.concatenate([x] * 3, axis=2)
    return x_, [mi,ma]

def tensor_to_img(tensor):
    to_img = T.ToPILImage()
    img = to_img(tensor)
    return img

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)

def visualize_latent(latent):
        if latent.shape == torch.Size([64,64,4]):
            latent = einops.rearrange(latent, 'h w c -> 1 c h w', c=4, h=64)
        elif latent.shape != torch.Size([1,4,64,64]):
            latent = einops.rearrange(latent, '(b h w) c -> b c h w', b=1, c=4, h=64)
        elif latent.shape == torch.Size([1,4,64,64]):
            latent=latent
        linear_decoding = [
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
        ]
        linear_decoding = torch.tensor(linear_decoding) #.to(device=torch_device) torch.Size([4, 3])
        linear_decoding = linear_decoding.to(device=latent.device)
        latent_to_img = torch.einsum('n c h w, c r -> n r h w', latent, linear_decoding)  # result1: [1,3,64,64]
        latent_to_img = (latent_to_img-latent_to_img.min())/(latent_to_img.max()-latent_to_img.min())
        latent_to_img_tensor = latent_to_img
        tk = tfms.ToPILImage()
        latent_to_img = tk(latent_to_img[0])
        return latent_to_img, latent_to_img_tensor

def visualize_grad(latent):
        if latent.shape == torch.Size([64,64,4]):
            latent = einops.rearrange(latent, 'h w c -> 1 c h w', c=4, h=64)
        elif latent.shape != torch.Size([1,4,64,64]):
            latent = einops.rearrange(latent, '(b h w) c -> b c h w', b=1, c=4, h=64)
        elif latent.shape == torch.Size([1,4,64,64]):
            latent=latent
        linear_decoding = [
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
        ]
        linear_decoding = torch.tensor(linear_decoding) #.to(device=torch_device) torch.Size([4, 3])
        linear_decoding = linear_decoding.to(device=latent.device)

        latent_to_img = torch.einsum('n c h w, c r -> n r h w', latent, linear_decoding)  # result1: [1,3,64,64]
        latent_to_img = (latent_to_img-latent_to_img.min())/(latent_to_img.max()-latent_to_img.min())

        latent_sum = torch.sum(latent, dim=(0,1))
        latent_sum = (latent_sum-latent_sum.min())/(latent_sum.max()-latent_sum.min())
        latent_mean = torch.mean(latent, dim=(0,1))
        latent_mean = (latent_mean-latent_mean.min())/(latent_mean.max()-latent_mean.min())

        latent_to_img_tensor = latent_to_img
        return latent_to_img_tensor, latent_sum, latent_mean

def visualize_mask(mask):
        mask = mask.to(torch.float)
        mask_tensor = mask
        tk = tfms.ToPILImage()
        mask_img = tk(mask[0])
        return mask_img, mask_tensor

__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim

def project_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]

    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_ref.device),
                                   torch.arange(0, width, dtype=torch.float32, device=depth_ref.device)])
    y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
    y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

    pts = torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(0) * (depth_ref.view(batchsize, -1).unsqueeze(1))

    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), pts)

    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                           torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize, 1, 1)), dim=1))[:, :3, :]

    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)  # B*3*20480
    depth_src = K_xyz_src[:, 2:3, :]
    xy_src = K_xyz_src[:, :2, :] / (K_xyz_src[:, 2:3, :] + 1e-9)
    x_src = xy_src[:, 0, :].view([batchsize, height, width])
    y_src = xy_src[:, 1, :].view([batchsize, height, width])

    return x_src, y_src, depth_src
# (x, y) --> (xz, yz, z) -> (x', y', z') -> (x'/z' , y'/ z')

def forward_warp(data, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
    x_res, y_res, depth_src = project_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src)
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]
    data = data[0].permute(1, 2, 0)
    new = torch.zeros_like(data)
    depth_src = depth_src.reshape(height, width)
    new_depth = torch.zeros_like(depth_src)
    yy_base, xx_base = torch.meshgrid([torch.arange(
        0, height, dtype=torch.long, device=depth_ref.device), torch.arange(0, width, dtype=torch.long, device=depth_ref.device)])
    y_res = torch.clip(y_res, 0, height - 1).to(torch.int64)
    x_res = torch.clip(x_res, 0, height - 1).to(torch.int64)
    yy_base = yy_base.reshape(-1)
    xx_base = xx_base.reshape(-1)
    y_res = y_res.reshape(-1)
    x_res = x_res.reshape(-1)
    # painter's algo
    for i in range(yy_base.shape[0]):
        if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
            new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
            new[y_res[i], x_res[i]] = data[yy_base[i], xx_base[i]]
    return new, new_depth

def forward_warp_backup(data, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
    x_res, y_res, depth_src = project_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src)
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]
    data = data[0].permute(1, 2, 0)
    new = np.zeros_like(data)
    depth_src = depth_src.reshape(height, width)
    new_depth = np.zeros_like(depth_src)
    yy_base, xx_base = torch.meshgrid([torch.arange(
        0, height, dtype=torch.long, device=depth_ref.device), torch.arange(0, width, dtype=torch.long)])
    y_res = np.clip(y_res.numpy(), 0, height - 1).astype(np.int64)
    x_res = np.clip(x_res.numpy(), 0, width - 1).astype(np.int64)
    yy_base = yy_base.reshape(-1)
    xx_base = xx_base.reshape(-1)
    y_res = y_res.reshape(-1)
    x_res = x_res.reshape(-1)
    # painter's algo
    for i in range(yy_base.shape[0]):
        if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
            new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
            new[y_res[i], x_res[i]] = data[yy_base[i], xx_base[i]]
    return new, new_depth


import torch.nn as nn
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
