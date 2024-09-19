import configargparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'TensorCP'])

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'own_data', 'llff_latent_mask'])


    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='learning rate')
    parser.add_argument("--lr_init_den", type=float, default=0.04,
                        help='learning rate')    
    parser.add_argument("--lr_init_app", type=float, default=0.02,
                        help='learning rate')        
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--entropy_weight", type=float, default=0.0,   # 0.0 #8e-5
                        help='loss weight')

    # mask threshold
    parser.add_argument("--acc_mask_threshold", type=float, default=0.0,   # 0.0 #8e-5
                        help='loss weight')

    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
                        
    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_Fea",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    


    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--render_all", type=int, default=0)
    parser.add_argument("--render_all_mask", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')



    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=100**3)
    parser.add_argument('--N_voxel_final',
                        type=int,
                        default=300**3)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument('--idx_view',
                        type=int,
                        default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')

    parser.add_argument("--cal_feature_up", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--feature_lambda", type=str, default='0.1')
    # parser.add_argument("--edit_config", type=str, default='query.yaml')
    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--save_freq", type=int, default=None, help='model weight save')
    # parser.add_argument("--using64x64", type=int, default=None, help='using 64x64 flag')
    parser.add_argument("--depth_ckpt", type=str, default=None, help='specific denisty weights npy file to reload for coarse network')
    parser.add_argument("--depth_KL_ckpt", type=str, default=None, help='specific denisty weights npy file to reload for coarse network')
    parser.add_argument("--eval_path_no_ndc", type=int, default=0)
    parser.add_argument("--no_ray_random", type=bool, default=False, help='when this switch in on, selecting ray no random')
    parser.add_argument("--render_latents", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", help='select device')
    parser.add_argument("--target_prompt", type=str, help='Editing NeRF Rendering result from taget prompt')
    parser.add_argument("--ref_prompt", type=str,default= '', help='Editing NeRF Rendering result from taget prompt')
    parser.add_argument("--ldm_device", type=str, default = 'cuda', help='Editing NeRF Rendering result from taget prompt')
    parser.add_argument("--mask_path", type=str, default = './data/nerf_llff_data/room_test/prompts/television/mask', help='select mask path')
    parser.add_argument("--mask_prompt", type=str, default = None, help='select mask path')
    parser.add_argument("--recon_weight", type=float, default=10.0, help='KL divergence loss weight')
    parser.add_argument("--sds_weight", type=float, default=1.0, help='gradient weight')
    parser.add_argument("--dds_weight", type=float, default=1.0, help='gradient weight')
    parser.add_argument("--im_recon_weight", type=float, default=0.01, help='gradient weight')
    parser.add_argument("--om_recon_weight", type=float, default=100.0, help='gradient weight')
    parser.add_argument("--depth_weight", type=float, help='depth_weight')
    parser.add_argument("--om_recon_w_init", type=float, default=1.0, help='gradient weight')
    parser.add_argument("--threshold_rate", type=float, default=0.01, help='threshold_rate')
    parser.add_argument("--max_step", type=int, default=980, help='max_step')
    # parser.add_argument("--anneal_thre", type=int, help='max_step')
    parser.add_argument("--cfg_init", type=float,default=10, help='cfg_init')
    parser.add_argument("--cfg_final", type=float,default=10, help='cfg_final')
    parser.add_argument("--wandb",  action='store_true')
    parser.add_argument("--i_wandb", type=int, default=10, help='frequency of logging on wandb(iteration)')
    parser.add_argument("--img_wandb", type=int, default=100, help='frequency of logging on wandb(iteration)')
    parser.add_argument("--attn_mask_use", type=int, default=0, help='select attention mask use iteration')
    parser.add_argument("--dds_mask_use", type=int, default=500, help='select attention mask use iteration')
    parser.add_argument("--prompt_index", type=str, default=-1, help='select prompt index for attn mask')
    parser.add_argument("--time_anneal", type=int, default=10000, help='select prompt index for attn mask')
    parser.add_argument("--time_anneal_max", type=int, default=980, help='select prompt index for attn mask')
    parser.add_argument("--what_mask", type=str, default='ref', help='select prompt index for attn mask')
    parser.add_argument("--ablation_list", type=str, help="List of ablation list")
    parser.add_argument("--corr_weight", type=float, help="Weight of correlation loss")
    parser.add_argument("--inv_mask_weight", type=float, default=0.3, help="Weight of correlation loss")
    parser.add_argument("--mask_ckpt", type=str, help="Pre-trained voxel mask attn")

    parser.add_argument("--lr_small_vae", type=float, default=0.02, help="learnining rate of small vae")
    parser.add_argument("--vae_lambda", type=float, default=1.0, help="learnining rate of small vae")
    parser.add_argument("--vae_lr", type=float, default=0.005, help="learnining rate of small vae")
    parser.add_argument("--svae_ckpt", type=str, default=None, help="checkpoint path of small vae")
    parser.add_argument("--use_temporal_attn", default=False, action='store_true',  help="Determine using temporal attention")
    parser.add_argument("--temporal_attn_start", type=int, default=5000, help="Determine using temporal attention")
    # parser.add_argument("--edit_iter", type=int, default=10, help='threshold_rate')

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()