import numpy as np
import torch
from run_nerf_helpers import *

def get_freq_reg_mask(dim, pos_enc_length, current_iter, reg_start_iter, reg_end_iter, min_visible=1, max_visible=None):
    
    """
    returns a frequency mask for positioning encoding in NSFF
    Args:
        dim : 3 or 4
        pose_enc_lenght
        current_iter
        start_iter
        total_reg_iter
        min_visible : (int) 1, 5, 9, Minimux visible range of the mask. Default may be 9 #FIXME
        max_visible : (float)
        
    Returns:
        mask!
    """
    
    if max_visible is None:
        
        if current_iter > reg_end_iter:
            return torch.ones(pos_enc_length).cuda().float()
        
        elif current_iter < reg_start_iter:
            freq_mask = np.zeros(pos_enc_length) # [84, ] or [27, ]
            ptr = min_visible
            int_ptr=int(ptr)
            freq_mask[: int_ptr * dim] = 1.0
            return torch.from_numpy(np.clip(np.array(freq_mask), 1e-8, 1 - 1e-8)).cuda().float()
        
        else:
            freq_mask = np.zeros(pos_enc_length) # [84, ] or [27, ]
            ptr = (pos_enc_length / dim - min_visible) * (current_iter-reg_start_iter) / (reg_end_iter-reg_start_iter) + min_visible
            ptr = ptr if ptr < pos_enc_length / dim else pos_enc_length / dim
            int_ptr = int(ptr)
            freq_mask[: int_ptr * dim] = 1.0
            freq_mask[int_ptr * dim : int_ptr * dim + dim] = (ptr - int_ptr)
            return torch.from_numpy(np.clip(np.array(freq_mask), 1e-8, 1 - 1e-8)).cuda().float()
        
    else:
        # For the ablation study that controls the maximum visible range of frequency spectrum
        freq_mask = np.zeros(pos_enc_length)
        freq_mask[: int(pos_enc_length * max_visible)] = 1.0 # e.g., max_visible: 0.43 
        return torch.from_numpy(np.array(freq_mask)).cuda().float()
        
    
def get_freq_mask_view(embedded_shape, current_iter, total_iter):
    if len(embedded_shape) == 2:
        embedded_shape = embedded_shape[1]
    if current_iter > total_iter:
        return torch.ones(embedded_shape).cuda().float()
    
    else:
        alpha_t = np.zeros(embedded_shape) # [27, ]
        ptr = embedded_shape / 3 * current_iter / total_iter + 1
        ptr = ptr if ptr < embedded_shape / 3 else embedded_shape / 3
        int_ptr = int(ptr)
        alpha_t[: int_ptr * 3] = 1.0
        alpha_t[int_ptr * 3 : int_ptr * 3 + 3] = (ptr - int_ptr)
        return torch.from_numpy(np.clip(np.array(alpha_t), 1e-8, 1 - 1e-8)).cuda().float()

def get_highfreq_embed():
    return get_embedder(multires=10, i=0, input_dims=4)