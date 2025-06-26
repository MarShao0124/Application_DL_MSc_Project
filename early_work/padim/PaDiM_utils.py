# -*- coding: utf-8 -*-
# """
# main.py
#   Original Author: 2021.05.02. @chanwoo.park
#   Edited by MarShao0124 2025.03.09
#   Edit: Correct the embbeding_concat function as described in the paper,
#   , where the output of the layers are concatenated by the w and h layer 3, largest layer in the depth. 

#   run PaDiM algorithm
#   Reference:
#       Defard, Thomas, et al. "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization."
#       arXiv preprint arXiv:2011.08785 (2020).
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os

# 2. Third-party modules
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from skimage import morphology
from skimage.segmentation import mark_boundaries

# 3. Own modules


################
#   Definition #
################
def embedding_concat(l1, l2, l3):
    bs, h1, w1, c1 = l1.shape
    _, h2, w2, c2 = l2.shape
    _, h3, w3, c3 = l3.shape

    l1 = tf.image.resize(l1, (h3, w3), method='bilinear')
    l2 = tf.image.resize(l2, (h3, w3), method='bilinear')
    embedding = tf.concat([l1, l2, l3], axis=-1)
    embedding = tf.reshape(embedding, (bs, h3 * w3, c1 + c2 + c3))

    return embedding


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)

    for i in range(num):
        # Handle test image
        img = test_img[i]
        if img.shape[0] == 3:  # If channels first, transpose to channels last
            img = img.transpose(1, 2, 0)

        # Handle ground truth mask
        gt = gts[i]
        if gt.ndim == 3:  # If 3D tensor
            gt = gt.squeeze()  # Remove singleton dimensions
        
        # Create heatmap and binary mask
        heat_map = scores[i] * 255
        # Ensure heat_map is 2D
        if heat_map.ndim == 3:
            heat_map = heat_map.squeeze()
        
        mask = scores[i].copy()
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask.squeeze()
            
        # Thresholding
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0

        # Apply morphological operations
        kernel = morphology.disk(4)
        # Ensure both mask and kernel are 2D
        mask = mask.astype(np.float32)  # Convert to float32 for morphological operations
        mask = morphology.opening(mask, kernel)
        mask *= 255
        
        # Create visualization
        # Ensure mask is in the correct format for mark_boundaries
        mask_2d = mask.astype(bool)  # Convert to boolean for mark_boundaries
        vis_img = mark_boundaries(img, mask_2d, color=(1, 0, 0), mode='thick')
        
        # Create subplot
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Configure axes
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
            
        # Plot original image
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        
        # Plot ground truth
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        
        # Plot heatmap
        # First show the base image in grayscale
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        # Then overlay the heatmap
        heatmap_overlay = ax_img[2].imshow(heat_map, cmap='jet', norm=norm, alpha=0.5)
        ax_img[2].title.set_text('Predicted heat map')
        
        # Plot predicted mask
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        
        # Plot segmentation
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation: {:.2f}'.format(img_scores[i]))
        
        # Add colorbar
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(heatmap_overlay, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        # Save and close
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def draw_auc(fp_list, tp_list, auc, path):
    plt.figure()
    plt.plot(fp_list, tp_list, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(path)

    plt.clf()
    plt.cla()
    plt.close()


def draw_precision_recall(precision, recall, base_line, path):
    f1_score = []
    for _idx in range(0, len(precision)):
        _precision = precision[_idx]
        _recall = recall[_idx]

        if _precision + _recall == 0:
            _f1 = 0
        else:
            _f1 = 2 * (_precision * _recall) / (_precision + _recall)
        f1_score.append(_f1)

    plt.figure()
    plt.plot(recall, precision, marker='.', label='precision-recall curve')
    plt.plot([0, 1], [base_line, base_line], linestyle='--', color='grey', label='No skill ({:.04f})'.format(base_line))
    plt.plot(recall, f1_score, linestyle='-', color='red', label='f1 score (Max.: {:.4f})'.format(np.max(f1_score)))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(path)

    plt.clf()
    plt.cla()
    plt.close()

    return np.max(f1_score)


def save_result(path, category, net_type, batch_size, rd ,auc, patch_auc, f1, base_line, inference_time):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['Category', 'net_type', 'batch_size', 'rd', 'Image_ROCAUC', 'Patch_ROCAUC', 'F1', 'BaseLine', 'InferenceTime(s)'])
    else:
        df = pd.read_csv(path)

    new_row = pd.DataFrame([{'Category': category, 
                             'net_type': net_type,
                             'batch_size': batch_size,
                             'rd': rd,
                             'Image_ROCAUC': auc, 
                             'Patch_ROCAUC': patch_auc, 
                             'F1': f1, 
                             'BaseLine': base_line,
                             'InferenceTime(s)': inference_time}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)
