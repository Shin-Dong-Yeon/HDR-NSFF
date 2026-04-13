import configargparse

def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',

                        help='input data directory')
    parser.add_argument("--render_lockcam_slowmo", action='store_true', 
                        help='render fixed view + slowmo')
    parser.add_argument("--render_slowmo_bt", action='store_true', 
                        help='render space-time interpolation')

    parser.add_argument("--final_height", type=int, default=288, 
                        help='training image height, default is 512x288')
    
    # HDR-NSFF add-ons ##
    parser.add_argument("--use_posenc_mask", action='store_true',
                        help='use positional encoding mask')
    parser.add_argument("--sf_multires", type=int, default=0,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--debug", action='store_true', help='debugging mode')
    parser.add_argument("--aligned_depth", action='store_true', help='use aligned_depth instead midas')
    parser.add_argument("--disp_model", type=str, default='midas', help='options: midas / aligned / depthcrafter / depth-anything')
    parser.add_argument("--sparse", action='store_true', help='sparse input or else')
    parser.add_argument("--unit_sf_bidirection", action='store_true', help='change bidirection loss as unit bidirection loss')
    
    parser.add_argument("--half_res", action='store_true', default=True, help='option for blender dataset')
    parser.add_argument("--testkip", type=int, default=1, help='option for blender dataset')
    
    parser.add_argument("--render_tm", type=str, default='tm', help='options: tm, wb, no')
    parser.add_argument("--skip_sf_reg_loss", action='store_true', default=False, help='option for not using sf_reg_loss')
    parser.add_argument("--skip_sf_sm_loss", action='store_true', default=False, help='option for not using sf_sm_loss')
    parser.add_argument("--skip_prob_reg_loss", action='store_true', default=False, help='option for not using prob_reg_loss')
    parser.add_argument("--traj_flow", action='store_true', default=False, help='option for using traj_flow')
    parser.add_argument("--w_traj_flow", type=float, default=0.02, help='weights of traj_flow loss')
    parser.add_argument("--decay_traj_flow_w", action='store_true', default=False, help='decay traj_flow weights')
    parser.add_argument("--step_frame", type=int, default=1, help='step frame for loading data')
    parser.add_argument("--render_train", action='store_true', default=False, help='render train data')
    parser.add_argument("--share_crf", action='store_true', default=False, help='share crf')
    parser.add_argument("--share_wb", action='store_true', default=False, help='share just 3 wb')
    parser.add_argument("--log_scale", action='store_true', default=False, help='use log scale crf')
    parser.add_argument("--unit_loss", action='store_true', default=False, help='use unit loss')
    parser.add_argument("--w_crf", type=float, default=0.001, help='weights of crf loss')
    parser.add_argument("--w_unit", type=float, default=0.001, help='weights of unit loss')
    parser.add_argument("--ref_idx", type=int, default=0, help='reference index for crf')
    parser.add_argument("--disp_loss", type=str, default=None, help='options: depth-anything-o1 / depth-anything-o2')
    parser.add_argument("--no_chain_5frames", action='store_true', default=False, help='options: True / False')
    parser.add_argument("--chain_5frames_flow", action='store_true', default=False, help='optionsl Ture / False')
    parser.add_argument("--custom_times", type=int, default=None, help='render at just custom times')
    parser.add_argument("--leaky_th_h", type=float, default=0.9, help='leaky threshold high')
    parser.add_argument("--leaky_th_l", type=float, default=0.15, help='leaky threshold high')
    parser.add_argument("--sat_mask_0", action='store_true', default=False, help='optionsl Ture / False')
    parser.add_argument("--sat_mask_3", action='store_true', default=False, help='optionsl Ture / False')
    parser.add_argument("--viz_crf", action='store_true', default=False, help='optionsl Ture / False')
    parser.add_argument("--render_dy", action='store_true', default=False)
    parser.add_argument("--crf_monotonic", action='store_true', default=False)
    parser.add_argument("--crf_start_step", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--w_percept", type=float, default=0.01)
    parser.add_argument("--use_percept", action='store_true', default=False)
    parser.add_argument("--start_percept", type=int, default=149999)
    parser.add_argument("--render_interpolate", action='store_true', default=False)
    parser.add_argument("--decay_crf_w", action='store_true', default=False)
    parser.add_argument("--crf_stop_step", type=int, default=9999999)
    parser.add_argument("--render_lockcam", action='store_true', default=False)
    parser.add_argument("--render_lockcam_slowmo_full", action='store_true', default=False)
    parser.add_argument("--freq_reg_start_iter", type=int, default=0)
    parser.add_argument("--freq_reg_end_iter", type=int, default=None)
    parser.add_argument("--freq_reg_min_visible", type=int, default=None)
    parser.add_argument("--freq_reg_max_visible", type=float, default=None)
    parser.add_argument("--use_freq_reg", action='store_true', default=False)
    parser.add_argument("--use_freq_reg_rigid", action='store_true', default=False)
    parser.add_argument("--new_blending", action='store_true', default=False)
    parser.add_argument("--use_motion_mask_weights", action='store_true', default=False)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--use_difix", action='store_true', default=False)
    parser.add_argument("--difix_interval", type=int, default=2000)
    parser.add_argument("--difix_kview", type=int, default=8)
    parser.add_argument("--difix_kview_step", type=int, default=1)
    parser.add_argument("--difix_save_root", type=str, default="difix_views")
    parser.add_argument("--w_difix_render", type=float, default=1.)
    parser.add_argument("--difix_ver2", action='store_true', default=False)
    parser.add_argument("--difix_start_step", type=int, default=None)
    parser.add_argument("--difix_freeze_rigid", action='store_true', default=False)
    parser.add_argument("--balanced_sampling", action='store_true', default=False)
    parser.add_argument("--use_render_l1", action='store_true', default=False)
    parser.add_argument("--difix_gt_ratio", type=float, default=0.5, help='probability of GT ratio, [0.,1.]')
    parser.add_argument("--difix_freeze_cam", action='store_true', default=False)
    parser.add_argument("--difix_l1", action='store_true', default=False)
    parser.add_argument("--difix_use_depth", action='store_true', default=False)
    parser.add_argument("--difix_use_depth_weight", type=float, default=1.)
    parser.add_argument("--difix_use_depth_weight_explicit", type=float, default=0.)
    parser.add_argument("--difix_w_render", type=float, default=1.)
    parser.add_argument("--difix_gt_ref", action='store_true', default=False)
    parser.add_argument("--w_render", type=float, default=1.)
    parser.add_argument("--mask_warp", action='store_true', default=False)
    parser.add_argument("--difix_weights_map_ray", action='store_true', default=False)
    parser.add_argument("--render_l1_l2", action='store_true', help='use MSE + alpha*MAE mixed loss')
    parser.add_argument("--w_l1_l2", type=float, default=0.01, help='alpha coefficient for MAE term')
    parser.add_argument("--difix_use_mv_mask", action="store_true", default=False)
    parser.add_argument("--difix_inter_pose", action="store_true", default=False)
    parser.add_argument("--difix_gt_ref_ratio", type=float, default=1.)
    parser.add_argument("--eval_start_time", type=int, default=0)
    parser.add_argument("--w_crf_monotonic", type=float, default=0.001)
    parser.add_argument("--w_crf_smooth", type=float, default=0.001)
    parser.add_argument("--share_crf_lr_ratio", type=float, default=1.)

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*128, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*128, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_iters", type=int, default=250001,
                        help='how many iteration do you want to run')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_bt", action='store_true', 
                        help='render bullet time')

    parser.add_argument("--render_test", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    parser.add_argument("--target_idx", type=int, default=10, 
                        help='target_idx')
    parser.add_argument("--num_extra_sample", type=int, default=512, 
                        help='num_extra_sample')
    parser.add_argument("--decay_depth_w", action='store_true', 
                        help='decay depth weights')
    parser.add_argument("--use_motion_mask", action='store_true', 
                        help='use motion segmentation mask for hard-mining data-driven initialization')
    parser.add_argument("--decay_optical_flow_w", action='store_true', 
                        help='decay optical flow weights')

    parser.add_argument("--w_depth",   type=float, default=0.04, 
                        help='weights of depth loss')
    parser.add_argument("--w_optical_flow", type=float, default=0.02, 
                        help='weights of optical flow loss')
    parser.add_argument("--w_sm", type=float, default=0.1, 
                        help='weights of scene flow smoothness')
    parser.add_argument("--w_sf_reg", type=float, default=0.1, 
                        help='weights of scene flow regularization')
    parser.add_argument("--w_cycle", type=float, default=0.1, 
                        help='weights of cycle consistency')
    parser.add_argument("--w_prob_reg", type=float, default=0.1, 
                        help='weights of disocculusion weights')

    parser.add_argument("--w_entropy", type=float, default=1e-3, 
                        help='w_entropy regularization weight')

    parser.add_argument("--decay_iteration", type=int, default=50, 
                        help='data driven priors decay iteration * 1000')

    parser.add_argument("--chain_sf", action='store_true', 
                        help='5 frame consistency if true, \
                             otherwise 3 frame consistency')

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=50)

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=10000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=50000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--use_tone_mapping", action='store_true', default=False, 
                        help='use tone mapping module')
    parser.add_argument("--use_tone_mapping_camparam", action='store_true', default=False,
                        help='for test in rendering, should be merge use_tone_mapping')
    parser.add_argument("--use_initialize", action='store_true', default=False, 
                        help='use tone mapping module')
    parser.add_argument('--use_sat_mask', action='store_true', default=False)
    parser.add_argument('--tone_mapping', type=str, choices=["piece_wise", "nn", "hdr_hexplane"], default = "piece_wise")
    parser.add_argument('--render_mode', type=str, default='full',
                        help='full / cb / dynamic /rigid')
    return parser