"""
inference_flow_01.py

Runs forward and backward semantic flow inference using a trained DINO-tracker model.
Outputs are saved to <data-path>/dino-tracker/trajectory_flow/ as:
  traj_flows_i{interval}_fwd.npy   (T-interval, H, W, 2)
  traj_flows_i{interval}_bwd.npy   (T-interval, H, W, 2)

This file lives in nsff_scripts/ and is copied to dino-tracker-temp/dino-tracker/
by run_dino-tracker_0.sh before execution.
"""
import os
import numpy as np
import torch
import argparse
from dino_tracker import DINOTracker
from models.model_inference import ModelInference
from data.data_utils import get_grid_query_points

device = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run(args):
    dino_tracker = DINOTracker(args)
    dino_tracker.load_fg_masks()
    model = dino_tracker.get_model()

    if args.iter is not None:
        model.load_weights(args.iter)

    traj_flow_dir = os.path.join(dino_tracker.grid_trajectories_dir, '..', 'trajectory_flow')
    os.makedirs(traj_flow_dir, exist_ok=True)

    model_inference = ModelInference(
        model=model,
        range_normalizer=dino_tracker.range_normalizer,
        anchor_cosine_similarity_threshold=dino_tracker.config['anchor_cosine_similarity_threshold'],
        cosine_similarity_threshold=dino_tracker.config['cosine_similarity_threshold'],
    )

    interval = args.interval

    # Forward flow
    print(f"Computing forward semantic flow with interval={interval} ...")
    fwd_flows = model_inference.compute_framewise_flow_tensor(
        dino_tracker.fg_masks, batch_size=10000, interval=interval)
    fwd_path = os.path.join(traj_flow_dir, f'traj_flows_i{interval}_fwd.npy')
    np.save(fwd_path, torch.stack(fwd_flows).cpu().detach().numpy())
    print(f"Saved: {fwd_path}")

    # Backward flow
    print(f"Computing backward semantic flow with interval={interval} ...")
    bwd_flows = model_inference.compute_framewise_flow_tensor_bwd(
        dino_tracker.fg_masks, batch_size=10000, interval=interval)
    bwd_path = os.path.join(traj_flow_dir, f'traj_flows_i{interval}_bwd.npy')
    np.save(bwd_path, torch.stack(bwd_flows).cpu().detach().numpy())
    print(f"Saved: {bwd_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/train.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/libby", type=str)
    parser.add_argument("--iter", type=int, default=None,
                        help="Checkpoint iteration to load. If None, loads the latest.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--interval", type=int, default=1,
                        help="Frame interval for flow computation.")
    parser.add_argument("--use-segm-mask", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=None)

    args = parser.parse_args()
    run(args)
