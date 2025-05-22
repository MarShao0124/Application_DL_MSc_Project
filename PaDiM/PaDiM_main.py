# -*- coding: utf-8 -*-
"""
new_padim_main.py
Main script to run the PyTorch implementation of PaDiM algorithm.
Reference:
    Defard, Thomas, et al. "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization."
    arXiv preprint arXiv:2011.08785 (2020).
"""

import os
import argparse
import random
import numpy as np
import torch
from PaDiM import padim

# For reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Define available targets for MVTec dataset
MVTEC_TARGETS = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of PaDiM')
    
    # Dataset parameters
    parser.add_argument('--target', type=str, default='bottle',
                        choices=MVTEC_TARGETS,
                        help='Target object from MVTec dataset')
    parser.add_argument('--data_path', type=str, 
                        default='data/mvtec_anomaly_detection',
                        help='Path to the MVTec dataset')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'wide_resnet50_2'],
                        help='Backbone network architecture')
    parser.add_argument('--rd', type=int, default=100,
                        help='Random sampling dimension')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Visualization parameters
    parser.add_argument('--save_plots', action='store_true',
                        help='Save visualization plots')
    parser.add_argument('--results_path', type=str, default='results',
                        help='Path to save results')
    
    # Run mode
    parser.add_argument('--run_all', action='store_true',
                        help='Run PaDiM on all MVTec categories')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_path, exist_ok=True)
    
    if args.run_all:
        # Run PaDiM on all MVTec categories
        results = {}
        for target in MVTEC_TARGETS:
            print(f"\n=== Running PaDiM on {target} ===")
            img_auc, pixel_auc = padim(
                category=target,
                batch_size=args.batch_size,
                rd=args.rd,
                is_plot=args.save_plots
            )
            results[target] = {
                'image_auc': img_auc,
                'pixel_auc': pixel_auc
            }
        
        # Print summary of results
        print("\n=== Summary of Results ===")
        img_aucs = []
        pixel_aucs = []
        for target, metrics in results.items():
            img_aucs.append(metrics['image_auc'])
            pixel_aucs.append(metrics['pixel_auc'])
            print(f"{target}:")
            print(f"  Image AUC: {metrics['image_auc']:.4f}")
            print(f"  Pixel AUC: {metrics['pixel_auc']:.4f}")
        
        print("\nOverall Performance:")
        print(f"Mean Image AUC: {np.mean(img_aucs):.4f}")
        print(f"Mean Pixel AUC: {np.mean(pixel_aucs):.4f}")
    
    else:
        # Run PaDiM on single category
        print(f"\n=== Running PaDiM on {args.target} ===")
        img_auc, pixel_auc = padim(
            category=args.target,
            batch_size=args.batch_size,
            rd=args.rd,
            is_plot=args.save_plots
        )
        print(f"\nResults for {args.target}:")
        print(f"Image AUC: {img_auc:.4f}")
        print(f"Pixel AUC: {pixel_auc:.4f}")

if __name__ == "__main__":
    main()

#python padim/padim_main.py --run_all --batch_size 32 --rd 100 --save_plots