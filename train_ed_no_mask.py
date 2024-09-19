
import os
from tqdm.auto import tqdm
from opt import config_parser
import glob
from stable_diffusion_attn import *
import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from torchvision import transforms as tfms
from dataLoader import dataset_dict
from little_vae import *
import sys
import matplotlib.pyplot as plt
import einops
from crs_controller import *
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast
torch2img = lambda x: (transforms.ToPILImage()(x).convert("RGB"))
_rearrange = lambda x: (einops.rearrange(x, '(b h w) c -> b c h w', b=1, c=4, h=64))
_rearrange_mask = lambda x: (einops.rearrange(x, '(b h w) c -> b c h w', b=1, c=1, h=64))
_rearrange2 = lambda x: (einops.rearrange(x, '(h w)->h w', h=64))
_dearrange = lambda x: (einops.rearrange(x, 'b c h w -> (b h w) c', b=1, c=4, h=64))

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]
    def nextids_no_rand(self, iteration):
        self.curr = (iteration % (self.total//self.batch)) * self.batch
        self.ids = torch.arange(self.total)
        return self.ids[self.curr:self.curr+self.batch]
    def nextids_rand(self, iteration):
        self.random_idx = np.random.randint(0,(self.total//self.batch))
        self.curr = self.random_idx * self.batch
        self.ids = torch.arange(self.total)
        return self.ids[self.curr:self.curr+self.batch]

class SimpleSampler_no_rand:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    device = torch.device(args.device)
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, args.feature_dir, args.mask_path, split='test', downsample=args.downsample_train, is_stack=True)
    all_dataset = dataset(args.datadir, args.feature_dir, args.mask_path,split='all', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    small_vae = little_vae(4,512)
    if args.svae_ckpt is not None:
        svae_ckpt = torch.load(args.svae_ckpt, map_location=device)
        small_vae.load(svae_ckpt)
        small_vae = small_vae.to(device)

    if args.target_prompt is not None:
        target_prompt = '_'.join(args.target_prompt.split(' '))
    else:
        target_prompt = ' '
    args.expname = args.datadir.split('/')[-1] + '_' + target_prompt

    if args.ckpt is None:
        ckpt_list = glob.glob('./log/direction_prompt' + args.expname+'/*.th')
        ckpt_path = ckpt_list[-1]
    elif args.ckpt is not None:
        ckpt_path = args.ckpt
    
    if not os.path.exists(ckpt_path):
        print('the ckpt path does not exists!! or matching specs are not exists!!')
        return


    # ckpt = torch.load(args.ckpt, map_location=device)
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, args.feature_dir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation_vae(train_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation_vae(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, small_vae=small_vae)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_all:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_render_all', exist_ok=True)
        evaluation_vae(all_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_render_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, small_vae=small_vae)

    if args.render_latents:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_latents(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):
    device = torch.device(args.device)
    # init dataset
    # args.dataset_name = 'llff_latent'
    dataset = dataset_dict[args.dataset_name]

    mask_path = os.path.join(args.datadir, 'prompts', args.mask_prompt, 'mask')
    all_dataset = dataset(args.datadir, args.feature_dir, mask_path,split='all', downsample=args.downsample_train, is_stack=True)
    train_dataset = dataset(args.datadir, args.feature_dir, mask_path, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, args.feature_dir, mask_path, split='test', downsample=args.downsample_train, is_stack=True)
    
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray
    target_prompt = args.target_prompt
    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    cfg_scheduler = CosineDecayScheduler(args.n_iters, args.cfg_init, args.cfg_final)

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    config_path = logfolder + "/config_spec.txt"
    if os.path.exists(config_path):
        f = open(config_path, "w")
    else:
        f = open(config_path, "a")
    f.write(str(args).replace(',', '\n'))
    f.close()
    
    aabb = train_dataset.scene_bbox.to(device)

    args.N_voxel_init = args.N_voxel_init ** 3
    args.N_voxel_final = args.N_voxel_final ** 3
    
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args .alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)
    
    small_vae = little_vae(4,512)
    if args.svae_ckpt is not None:
        svae_ckpt = torch.load(args.svae_ckpt, map_location=device)
        small_vae.load(svae_ckpt)
        small_vae = small_vae.to(device)

    # tensorf.freeze_basis_mat_renderer()
    grad_vars = tensorf.get_optparam_groups(args.lr_init_den, args.lr_init_app, args.lr_basis)
    grad_vars += [{'params': small_vae.parameters(), 'lr':args.vae_lr}]
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]
    allmask = train_dataset.all_mask
    # print("allmask shape: ", allmask.shape)
    allrays, allrgbs, allfeatures = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_features  # [N*W*H,6], [N*W*H,3], [N,4,W,H]
    # allmasks = train_dataset.all_mask
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    args.feature_lambda = torch.tensor(float(args.feature_lambda))
    entropy_weight = args.entropy_weight
    print("initial entropy reg weight", entropy_weight)
    threshold = args.acc_mask_threshold
    print("initial acc_mask_threshold", threshold)
    print("sds_weight", args.sds_weight)
    print("om_recon_weight", args.om_recon_weight)
    print("im_recon_weight", args.im_recon_weight)
    tags = []

    #########################################
    # initialize the Stable diffusion model #
    #########################################
    model_name = 'CompVis/stable-diffusion-v1-4'
    diffusion_model = StableDiffusion(args.ldm_device, model_name=model_name)
    vae = diffusion_model.vae
    for p in diffusion_model.parameters():
        p.requires_grad = False

    #########################################
    # initialize the Stable diffusion model #
    #########################################
    target_prompt = args.target_prompt
    ref_prompt = args.ref_prompt
    tgt_emb = diffusion_model.get_text_embeds(target_prompt)
    ref_emb = diffusion_model.get_text_embeds(ref_prompt)
    threshold_rate = args.threshold_rate
    dds_weight = args.dds_weight
    max_step = args.max_step
    total_loss = 0
    min_step = 20
    max_step = 980
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    what_mask = args.what_mask
    if what_mask == 'tgt':
        mask_prompt = args.target_prompt
    elif what_mask == 'ref':
        mask_prompt = args.ref_prompt

    controller = AttentionStore()
    register_attention_control(diffusion_model.pipeline, controller)
    for iteration in pbar:
        total_loss = 0
        if args.no_ray_random is True:
            ray_idx = trainingSampler.nextids_rand(iteration)
        else:
            ray_idx = trainingSampler.nextids()

        optimizer.zero_grad()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
        mask_train = allmask[ray_idx]
        # print("mask_train shape: ", mask_train.shape)
        mask_train = ~mask_train
        mask_train = _rearrange_mask(mask_train)
        mask_img = torch2img(mask_train[0,:,:,:].to(torch.float))
        mask_train = torch.cat([mask_train]*4, dim=1)
        feature_train = allfeatures[ray_idx] # [N,4,W,H] -> [1, 4, W, H]
        # mask_train = allmasks[ray_idx]
        # core training process

        # mask_train = mask_train.to(device)
        feature_train = feature_train.to(device).detach()
        feature_map, alphas_map, depth_map, weights, acc_maps = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        feature_map = small_vae(_rearrange(feature_map))
        # print("feature_map after vae:", feature_map.shape)
        # print("feature_train shape: ", feature_train.shape)

        z = feature_map
        feature_map = _dearrange(feature_map)
        # print("feature map after _dearrage: ", feature_map.shape)

        # print("feature map type to optimize: ", type(feature_map))

        z = z.to(args.ldm_device)

        z_hat = _rearrange(feature_train)

        # print("z_hat shape: ", z_hat.shape)

        z_hat = z_hat.to(args.ldm_device)
        
        # if iteration >= args.time_anneal:

        if iteration >= args.time_anneal:
            max_step = args.time_anneal_max
            min_step = 20

        guidance_scale = cfg_scheduler.get_lr(iteration)

        dds, t_dds, tgt_grad, ref_grad, attn_img = diffusion_model.delta_denoising_score(controller, tgt_emb, ref_emb, z, z_hat, 
                                                        guidance_scale, min_step=min_step, max_step=max_step, prompt = mask_prompt, what_mask = what_mask)

        dds_norm = dds.norm()
        org_dds = dds.detach().cpu()

        # dds[mask_train] = 0
        dds = dds * dds_weight

        dds_vis, dds_sum, dds_mean = visualize_grad(dds)
        tgt_grad_vis, tgt_grad_sum, tgt_grad_mean = visualize_grad(tgt_grad)
        ref_grad_vis, ref_grad_sum, ref_grad_mean = visualize_grad(ref_grad)
        org_dds_vis, org_dds_sum, org_dds_mean = visualize_grad(org_dds)

        z.backward(gradient=dds, retain_graph=True)

        # print("mask shape: ",mask_train.shape)

        mask2 = _dearrange(mask_train).to(device)

        # print("mask2 shape:", mask2.shape)

        z = _dearrange(z).to(args.device)
        
        # print("feature map shape: ", feature_map.shape)

        # recon_loss = torch.mean((feature_map[mask2] - feature_train[mask2]) ** 2)  * args.om_recon_weight
        #inrecon_loss = torch.mean((feature_map[~mask2] - feature_train[~mask2]) ** 2)  * args.im_recon_weight
        recon_loss = torch.mean((feature_map - feature_train) ** 2)  * args.om_recon_weight
        inrecon_loss = torch.mean((feature_map - feature_train) ** 2)  * args.im_recon_weight
        total_loss += recon_loss + inrecon_loss
        
        total_loss.backward()
        optimizer.step()


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor
            
        if (iteration + 1) % args.save_freq == 0:
            tensorf.save(f'{logfolder}/{iteration+1}.th')    
            small_vae.save(f'{logfolder}/{iteration+1}_svae.th')    

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' recon_loss = {recon_loss:.6f}'
                + f' total_loss = {total_loss:.6f}'
            )
            PSNRs = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation_vae(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False, device=device, small_vae=small_vae)

    tensorf.save(f'{logfolder}/{args.expname}.th')
    small_vae.save(f'{logfolder}/{args.expname}_svae.th') 


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation_vae(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,small_vae=small_vae)

    if args.render_all:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_render_all', exist_ok=True)
        evaluation_vae(all_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_render_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,small_vae=small_vae)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20230913)
    np.random.seed(20230913)

    args = config_parser()
    print(args)
    
    if type(args.render_only==str):
        args.render_only = int(args.render_only)
    if type(args.render_test==str):
        args.render_test = int(args.render_test)
    if type(args.render_train==str):
        args.render_train = int(args.render_train)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

