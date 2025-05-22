import os, sys
import timm
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from scipy.spatial.distance import mahalanobis
import sklearn.metrics as metrics
from mvtec_loader import MVTecDataset

from PaDiM_utils import plot_fig, draw_auc, draw_precision_recall, save_result
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from timm_extractor import TimmFeatureExtractor
from anomaly_map import AnomalyMapGenerator

class MultiVariateGaussian(nn.Module):
    """Multi Variate Gaussian Distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("inv_covariance", torch.empty(0))
        self.mean: torch.Tensor
        self.inv_covariance: torch.Tensor

    @staticmethod
    def _cov(observations: torch.Tensor, rowvar: bool = False) -> torch.Tensor:
        """Estimate covariance matrix of the observations."""
        if observations.dim() == 1:
            observations = observations.view(-1, 1)
        if rowvar and observations.shape[0] != 1:
            observations = observations.t()
        
        avg = torch.mean(observations, dim=0)
        observations_m = observations - avg.unsqueeze(0)
        covariance = torch.matmul(observations_m.t(), observations_m)
        covariance = covariance / (observations.shape[0] - 1)
        
        return covariance.squeeze()

    def forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """Calculate multivariate Gaussian distribution parameters.
        
        Args:
            embedding: Input tensor of shape (B, C, H*W) containing feature embeddings.
        
        Returns:
            List containing mean tensor and inverse covariance tensor
        """
        device = embedding.device
        
        # Calculate mean over batch dimension
        self.mean = torch.mean(embedding, dim=0)  # (C, H*W)
        
        batch_size, num_features, num_elements = embedding.size()
        covariance = torch.zeros(num_elements, num_features, num_features, device=device)
        identity = torch.eye(num_features, device=device)
        
        # Calculate covariance for each position
        for i in range(num_elements):
            covariance[i] = self._cov(embedding[:, :, i]) + 0.01 * identity
            
        # Add small regularization term for stability
        covariance = covariance + 1e-5 * identity.unsqueeze(0)
        
        # Calculate inverse covariance
        if device.type == "mps":
            self.inv_covariance = torch.linalg.inv(covariance.cpu()).to(device)
        else:
            self.inv_covariance = torch.linalg.inv(covariance)
            
        return [self.mean, self.inv_covariance]

    def fit(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """Fit multivariate Gaussian distribution to input embeddings."""
        return self.forward(embedding)

def generate_embedding(features: dict[str, torch.Tensor], layers: list[str], rd_indices: torch.Tensor) -> torch.Tensor:
    """Generate embedding from hierarchical feature map.
    
    Args:
        features: Dictionary of feature tensors from different layers
        layers: List of layer names in order
        rd_indices: Indices for random dimension selection
    
    Returns:
        Concatenated and subsampled embedding tensor
    """
    embeddings = features[layers[0]]
    for layer in layers[1:]:
        layer_embedding = features[layer]
        layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
        embeddings = torch.cat((embeddings, layer_embedding), 1)
    
    # subsample embeddings
    return torch.index_select(embeddings, 1, rd_indices) 

def padim(category, batch_size=32, rd=100, is_plot=True):
    # Initialize dataset
    train_dataset = MVTecDataset(
        root_path='data/mvtec_anomaly_detection',
        category=category,
        is_train=True
    )
    test_dataset = MVTecDataset(
        root_path='data/mvtec_anomaly_detection',
        category=category,
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize feature extractor and distribution estimator
    feature_extractor = TimmFeatureExtractor(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )
    feature_extractor.eval()
    gaussian = MultiVariateGaussian()
    
    # Initialize anomaly map generator
    anomaly_map_generator = AnomalyMapGenerator(sigma=4)

    # Define layers in order and print their shapes
    layers = ["layer1", "layer2", "layer3"]

    # Get total feature dimensions from a sample batch
    sample_batch = next(iter(train_loader))[0]
    features = feature_extractor(sample_batch) # (b, c, h, w)
    total_features = sum(feat.shape[1] for feat in features.values())
    
    # Random selection of features
    rd_indices = torch.randperm(total_features)[:rd] 

    # Training phase - collect features and compute distribution
    train_outputs = []
    for batch_imgs, _, _ in train_loader:
        with torch.no_grad():
            features = feature_extractor(batch_imgs) 
            embedding = generate_embedding(features, layers, rd_indices)
            b, c, h, w = embedding.shape # (b, rd, h, w)
            train_outputs.append(embedding.reshape(b, c, -1)) # (b, rd, h*w)
    
    train_outputs = torch.cat(train_outputs, dim=0) # (train_size, rd, h*w)
    
    # Calculate distribution parameters using MultiVariateGaussian
    mean, inv_covariance = gaussian.fit(train_outputs)

    # Testing phase
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    scores = []
    
    for batch_imgs, batch_masks, batch_labels in test_loader:
        test_imgs.extend(batch_imgs.cpu().numpy())
        gt_list.extend(batch_labels.cpu().numpy())
        gt_mask_list.extend(batch_masks.cpu().numpy())
        
        with torch.no_grad():
            features = feature_extractor(batch_imgs)
            embedding = generate_embedding(features, layers, rd_indices)
            b, c, h, w = embedding.shape
            
            # Generate anomaly map
            score_map = anomaly_map_generator(
                embedding=embedding,
                mean=mean,
                inv_covariance=inv_covariance,
                image_size=(224, 224)
            )
            scores.extend(score_map.cpu().numpy())

    scores = np.array(scores)
    gt_mask_list = np.array(gt_mask_list)
    gt_list = np.array(gt_list)
    test_imgs = np.array(test_imgs)

    # Normalize scores
    max_score = scores.max()
    min_score = scores.min()
    scores = (scores - min_score) / (max_score - min_score)

    # Calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    img_roc_auc = metrics.roc_auc_score(gt_list, img_scores)
    
    # Calculate pixel-wise ROC AUC score
    pixel_roc_auc = metrics.roc_auc_score(gt_mask_list.flatten(), scores.flatten())
    
    if is_plot:
        # Calculate metrics for plotting
        fpr, tpr, _ = metrics.roc_curve(gt_list, img_scores)
        precision, recall, _ = metrics.precision_recall_curve(gt_list, img_scores)
        
        # Get optimal threshold
        precision_pixel, recall_pixel, threshold = metrics.precision_recall_curve(
            gt_mask_list.flatten(), scores.flatten(), pos_label=1
        )
        f1_scores = 2 * precision_pixel * recall_pixel / (precision_pixel + recall_pixel)
        optimal_threshold = threshold[np.argmax(f1_scores)]
        
        # Save plots
        save_dir = os.path.join(os.path.dirname(__file__), 'results', category)
        os.makedirs(save_dir, exist_ok=True)
        
        # Draw ROC and PR curves
        draw_auc(fpr, tpr, img_roc_auc, os.path.join(save_dir, f'AUROC-{category}.png'))
        base_line = np.sum(gt_list) / len(gt_list)
        f1 = draw_precision_recall(precision, recall, base_line, os.path.join(save_dir, f'PR-{category}.png'))
        
        # Plot sample results
        plot_fig(test_imgs, scores, gt_mask_list, optimal_threshold, save_dir, category)
        
        # Save results
        save_dir = os.path.join(os.path.dirname(__file__), 'results', 'metrics.csv')
        save_result(save_dir, category, 'resnet18', batch_size, rd, img_roc_auc, pixel_roc_auc, f1, base_line, 0)

    print(f'Category: {category}')
    print(f'Image AUC: {img_roc_auc:.4f}')
    print(f'Pixel AUC: {pixel_roc_auc:.4f}')
    
    return img_roc_auc, pixel_roc_auc

    



