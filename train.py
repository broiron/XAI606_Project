
import os
from tqdm.auto import tqdm
from opt import config_parser
import glob


import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from little_vae import *
from diffusers import AutoencoderKL

from dataLoader import dataset_dict
import sys
import matplotlib.pyplot as plt
import einops

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast
renderer_depth = OctreeRender_trilinear_fast_depth
_rearrange = lambda x: (einops.rearrange(x, '(b h w) c -> b c h w', b=1, c=4, h=64))
_dearrange = lambda x: (einops.rearrange(x, 'b c h w -> (b h w) c', b=1, c=4, h=64))

small_tensorf_dir = '/home/park/research/TensoRF_small/log/fortress_MLP_Fea_64_no_vfp_l1'    #fortress_MLP_Fea_64_no_vfp_l1 추가
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
    test_dataset = dataset(args.datadir, args.feature_dir,split='test', downsample=args.downsample_train, is_stack=True)
    all_dataset = dataset(args.datadir, args.feature_dir,split='all', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray
    
    if args.svae_ckpt is None:
        return NotImplementedError
    elif args.svae_ckpt is not None:
        svae_ckpt = torch.load(args.svae_ckpt, map_location=device)
        small_vae = little_vae(4,512)
        small_vae.load(svae_ckpt)
        small_vae = small_vae.to(device)

    if args.ckpt is None:
        ckpt_list = glob.glob('./log/' + args.expname+'/*.th')
        ckpt_path = ckpt_list[-1]
    elif args.ckpt is not None:
        ckpt_path = args.ckpt
    
    if not os.path.exists(ckpt_path):
        print('the ckpt path does not exists!! or matching specs are not exists!!')
        return


    # ckpt = torch.load(args.ckpt, map_location=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(ckpt_path)
    if args.render_train:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, args.feature_dir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation_vae(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, small_vae=small_vae)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_all:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_render_all', exist_ok=True)
        PSNRs_test = evaluation_vae(all_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_render_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, small_vae=small_vae)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')


    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_latents:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_latents(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):
    device = torch.device(args.device)
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, args.feature_dir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, args.feature_dir, split='test', downsample=args.downsample_train, is_stack=True)
    all_dataset = dataset(args.datadir, args.feature_dir,split='all', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    
    if len(args.n_lamb_sigma) == 1:
        n_lamb_sigma.append(args.n_lamb_sigma[0])
        n_lamb_sigma.append(args.n_lamb_sigma[0])
    
    if len(args.n_lamb_sh) == 1:
        n_lamb_sh.append(args.n_lamb_sh[0])
        n_lamb_sh.append(args.n_lamb_sh[0])

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


    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
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

    # grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    grad_vars = tensorf.get_optparam_groups(args.lr_init_den, args.lr_init_app, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    small_vae = little_vae(4,512)
    # small_vae.load_vae_weight(vae)
    small_vae = small_vae.to(device)
    grad_vars += [{'params': small_vae.parameters(), 'lr':args.vae_lr}]
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
    # optimizer = torch.optim.AdamW(grad_vars)
    print("use Adam!")
    # optimizer = torch.optim.SGD(grad_vars)

    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]
    PSNR_mine =[]

    allrays, allrgbs, allfeatures = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_features  # [N*W*H,6], [N*W*H,3], [N,4,W,H]
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
    total_loss_list = []
    

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        if args.no_ray_random is True:
            ray_idx = trainingSampler.nextids_no_rand(iteration)
        else:
            ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
        feature_train = allfeatures[ray_idx]

        feature_train = feature_train.to(device).detach()
        # print(rays_train.shape)
        feature_map, alphas_map, depth_map, weights, acc_maps = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        # print("training shape", feature_map.shape) # [4096, 4]

        mse_loss = torch.mean((torch.abs(feature_map - feature_train))) # L1 Loss
        vae_output = small_vae(_rearrange(feature_map))
        vae_output = _dearrange(vae_output)
        vae_loss = torch.mean((torch.abs(vae_output - feature_train)))
        origin_total_loss = mse_loss + vae_loss
        vae_loss_weight = args.vae_lambda

        if hasattr(args, 'feature_stop_iter'):
            if iteration >= args.feature_stop_iter:
                feature_loss_weight = 0
        else:
            feature_loss_weight = args.feature_lambda
        total_loss = mse_loss * feature_loss_weight + vae_loss * vae_loss_weight
        
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        mse_loss = mse_loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(mse_loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse_loss', mse_loss, global_step=iteration)
        summary_writer.add_scalar('train/vae_loss', vae_loss, global_step=iteration)
        summary_writer.add_scalar('train/lr_factor', lr_factor, global_step=iteration)

        if iteration % 100 == 0:
            total_loss_list.append(mse_loss)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor
            
        if (iteration + 1) % args.save_freq == 0:
            tensorf.save(f'{logfolder}/{iteration+1}.th')    
            small_vae.save(f'{logfolder}/{iteration+1}_svae.th')    

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' feature_loss_weight = {feature_loss_weight:.2f}'
                + f' mse_loss = {mse_loss:.6f}'
                + f' vae_loss = {vae_loss:.6f}'
                + f' total_loss = {total_loss:.6f}'
            )
            PSNRs = []
            #feature_loss_weight


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation_vae(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False, device=device,small_vae=small_vae)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters) #(args.lr_init_den, args.lr_init_app, args.lr_basis)
            # grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            grad_vars = tensorf.get_optparam_groups(args.lr_init_den*lr_scale, args.lr_init_app*lr_scale, args.lr_basis*lr_scale)
            grad_vars += [{'params': small_vae.parameters(), 'lr':args.vae_lr*lr_scale}]
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    tensorf.save(f'{logfolder}/{args.expname}.th')
    small_vae.save(f'{logfolder}/{args.expname}_svae.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        # PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
        #                         N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        PSNRs_test = evaluation_vae(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, small_vae=small_vae)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        PSNRs_test_mean = np.round(np.mean(PSNRs_test), decimals=4)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    
    if args.render_all:
        os.makedirs(f'{logfolder}/imgs_render_all', exist_ok=True)
        # evaluation(all_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_render_all/',
        #                         N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        evaluation_vae(all_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_render_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, small_vae=small_vae)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)
    
    if type(args.render_only==str):
        args.render_only = int(args.render_only)
    if type(args.render_test==str):
        args.render_test = int(args.render_test)
    if type(args.render_train==str):
        args.render_train = int(args.render_train)
    if type(args.render_path==str):
        args.render_path = int(args.render_path)

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

