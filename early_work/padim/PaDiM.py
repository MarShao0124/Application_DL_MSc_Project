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

class PaDiM:
    """PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization.
    
    This class implements the PaDiM algorithm for anomaly detection and localization.
    It uses a pre-trained CNN backbone to extract features and models the distribution
    of normal patches using multivariate Gaussian distributions.
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],
        rd: int = 100,
        batch_size: int = 32,
        sigma: int = 4
    ):
        """Initialize PaDiM model.
        
        Args:
            backbone: Name of the backbone network architecture
            layers: List of layer names to extract features from
            rd: Number of random dimensions to select
            batch_size: Batch size for training
            sigma: Sigma value for Gaussian smoothing of anomaly maps
        """
        self.backbone = backbone
        self.layers = layers
        self.rd = rd
        self.batch_size = batch_size
        self.sigma = sigma
        
        # Initialize components
        self.feature_extractor = TimmFeatureExtractor(
            backbone=backbone,
            layers=layers,
        )
        self.feature_extractor.eval()
        self.gaussian = MultiVariateGaussian()
        self.anomaly_map_generator = AnomalyMapGenerator(sigma=sigma)
        
        # Initialize state variables
        self.mean = None
        self.inv_covariance = None
        self.rd_indices = None
        
    def _get_total_features(self, sample_batch: torch.Tensor) -> int:
        """Get total number of features from all layers.
        
        Args:
            sample_batch: Sample batch of images
            
        Returns:
            Total number of features across all layers
        """
        features = self.feature_extractor(sample_batch)
        return sum(feat.shape[1] for feat in features.values())
    
    def _generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate embedding from hierarchical feature map.
        
        Args:
            features: Dictionary of feature tensors from different layers
            
        Returns:
            Concatenated and subsampled embedding tensor
        """
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)
        
        # subsample embeddings
        return torch.index_select(embeddings, 1, self.rd_indices)
    
    def fit(self, train_loader: DataLoader) -> None:
        """Fit the model to the training data.
        
        Args:
            train_loader: DataLoader containing training images
        """
        # Get total feature dimensions and select random dimensions
        sample_batch = next(iter(train_loader))[0]
        total_features = self._get_total_features(sample_batch)
        self.rd_indices = torch.randperm(total_features)[:self.rd]
        
        # Collect features and compute distribution
        train_outputs = []
        for batch_imgs, _, _ in train_loader:
            with torch.no_grad():
                features = self.feature_extractor(batch_imgs)
                embedding = self._generate_embedding(features)
                b, c, h, w = embedding.shape
                train_outputs.append(embedding.reshape(b, c, -1))
        
        train_outputs = torch.cat(train_outputs, dim=0)
        
        # Calculate distribution parameters
        self.mean, self.inv_covariance = self.gaussian.fit(train_outputs)
    
    def predict(self, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict anomaly scores for test images.
        
        Args:
            test_loader: DataLoader containing test images
            
        Returns:
            Tuple containing:
            - test_imgs: Original test images
            - scores: Anomaly score maps
            - gt_list: Ground truth labels
            - gt_mask_list: Ground truth masks
        """
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        scores = []
        
        for batch_imgs, batch_masks, batch_labels in test_loader:
            test_imgs.extend(batch_imgs.cpu().numpy())
            gt_list.extend(batch_labels.cpu().numpy())
            gt_mask_list.extend(batch_masks.cpu().numpy())
            
            with torch.no_grad():
                features = self.feature_extractor(batch_imgs)
                embedding = self._generate_embedding(features)
                
                # Generate anomaly map
                score_map = self.anomaly_map_generator(
                    embedding=embedding,
                    mean=self.mean,
                    inv_covariance=self.inv_covariance,
                    image_size=(224, 224)
                )
                scores.extend(score_map.cpu().numpy())
        
        # Convert to numpy arrays
        scores = np.array(scores)
        gt_mask_list = np.array(gt_mask_list)
        gt_list = np.array(gt_list)
        test_imgs = np.array(test_imgs)
        
        # Normalize scores
        max_score = scores.max()
        min_score = scores.min()
        scores = (scores - min_score) / (max_score - min_score)
        
        return test_imgs, scores, gt_list, gt_mask_list
    
    def evaluate(self, scores: np.ndarray, gt_list: np.ndarray, gt_mask_list: np.ndarray) -> tuple[float, float]:
        """Evaluate model performance.
        
        Args:
            scores: Anomaly score maps
            gt_list: Ground truth labels
            gt_mask_list: Ground truth masks
            
        Returns:
            Tuple containing image-level and pixel-level ROC AUC scores
        """
        # Calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        img_scores_binary = np.where(img_scores >= 0.1, 1, 0)
        img_roc_auc = metrics.roc_auc_score(gt_list, img_scores_binary)

        
        # Calculate pixel-wise ROC AUC score
        pixel_roc_auc = metrics.roc_auc_score(gt_mask_list.flatten(), scores.flatten())
        
        return img_roc_auc, pixel_roc_auc
    
    def run(self, category: str, is_plot: bool = True) -> tuple[float, float]:
        """Run the complete PaDiM pipeline.
        
        Args:
            category: Category name from MVTec dataset
            is_plot: Whether to generate and save plots
            
        Returns:
            Tuple containing image-level and pixel-level ROC AUC scores
        """
        # Initialize datasets
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
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # Fit model
        self.fit(train_loader)
        
        # Get predictions
        test_imgs, scores, gt_list, gt_mask_list = self.predict(test_loader)
            
        # Evaluate performance
        img_roc_auc, pixel_roc_auc = self.evaluate(scores, gt_list, gt_mask_list)
        
        if is_plot:
            # Calculate metrics for plotting
            fpr, tpr, _ = metrics.roc_curve(gt_list, scores.reshape(scores.shape[0], -1).max(axis=1))
            precision, recall, _ = metrics.precision_recall_curve(gt_list, scores.reshape(scores.shape[0], -1).max(axis=1))
            
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
            save_result(save_dir, category, self.backbone, self.batch_size, self.rd, img_roc_auc, pixel_roc_auc, f1, base_line, 0)
        
        return img_roc_auc, pixel_roc_auc

# For backward compatibility
def padim(category, batch_size=32, rd=100, is_plot=True):
    """Legacy function for backward compatibility."""
    model = PaDiM(batch_size=batch_size, rd=rd)
    return model.run(category, is_plot)

    



