# -*- coding: utf-8 -*-
"""
new_padim_main.py
Main script to run the PyTorch implementation of PaDiM algorithm.
Reference:
    Defard, Thomas, et al. "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization."
    arXiv preprint arXiv:2011.08785 (2020).
"""

import os
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

def main():
    # Configuration
    seed = 42
    results_path = 'results'
    run_all = True  # Set to True to run all categories
    target = 'bottle'  # Only used if run_all=False
    batch_size = 32
    rd = 100
    save_plots = True

    # Set random seed for reproducibility
    set_seed(seed)
    
    # Create results directory if it doesn't exist
    results_path = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_path, exist_ok=True)
    
    if run_all:
        # Run PaDiM on all MVTec categories
        results = {}
        for target in MVTEC_TARGETS:
            print(f"\n=== Running PaDiM on {target} ===")
            img_auc, pixel_auc = padim(
                category=target,
                batch_size=batch_size,
                rd=rd,
                is_plot=save_plots
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
        print(f"\n=== Running PaDiM on {target} ===")
        img_auc, pixel_auc = padim(
            category=target,
            batch_size=batch_size,
            rd=rd,
            is_plot=save_plots
        )
        print(f"\nResults for {target}:")
        print(f"Image AUC: {img_auc:.4f}")
        print(f"Pixel AUC: {pixel_auc:.4f}")

if __name__ == "__main__":
    main()