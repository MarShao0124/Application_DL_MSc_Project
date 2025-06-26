import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, root_path='data/mvtec_anomaly_detection', category='bottle', is_train=True, resize=256, cropsize=224):
        self.root_path = root_path
        self.category = category
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        
        # Define the category labels
        self.category_labels = {
            'bottle': ['good', 'broken_large', 'broken_small', 'contamination'],
            'cable': ['good', 'bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation', 
                     'cut_outer_insulation', 'missing_cable', 'missing_wire', 'poke_insulation'],
            'capsule': ['good', 'crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze'],
            'carpet': ['good', 'color', 'cut', 'hole', 'metal_contamination', 'thread'],
            'grid': ['good', 'bent', 'broken', 'glue', 'metal_contamination', 'thread'],
            'hazelnut': ['good', 'crack', 'cut', 'hole', 'print'],
            'leather': ['good', 'color', 'cut', 'fold', 'glue', 'poke'],
            'metal_nut': ['good', 'bent', 'color', 'flip', 'scratch'],
            'pill': ['good', 'color', 'combined', 'contamination', 'crack', 'faulty_imprint', 
                    'pill_type', 'scratch'],
            'screw': ['good', 'manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top'],
            'tile': ['good', 'crack', 'glue_strip', 'gray_stroke', 'oil', 'rough'],
            'toothbrush': ['good', 'defective'],
            'transistor': ['good', 'bent_lead', 'cut_lead', 'damaged_case', 'misplaced'],
            'wood': ['good', 'color', 'combined', 'hole', 'liquid', 'scratch'],
            'zipper': ['good', 'broken_teeth', 'combined', 'fabric_border', 'fabric_interior', 
                      'rough', 'split_teeth', 'squeezed_teeth']
        }

        # Setup transforms
        
        self.transform_x = transforms.Compose([
            transforms.ToTensor()])
        """
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
                              ])"""

        # Get image files and labels
        self.x_paths, self.y_paths, self.labels = self._get_image_files()

    def _get_image_files(self):
        x_paths = []
        y_paths = []  # For mask paths
        labels = []   # Binary labels: 0 for normal, 1 for anomaly

        if self.is_train:
            # During training, only load 'good' images
            img_dir = os.path.join(self.root_path, self.category, 'train/good')
            img_files = os.listdir(img_dir)
            x_paths.extend([os.path.join(img_dir, f) for f in img_files])
            y_paths.extend([None] * len(img_files))  # No masks for good samples
            labels.extend([0] * len(img_files))      # 0 for normal samples
        else:
            # During testing, load both good and defective images
            for label in self.category_labels[self.category]:
                img_dir = os.path.join(self.root_path, self.category, f'test/{label}')
                if not os.path.exists(img_dir):
                    continue
                
                img_files = os.listdir(img_dir)
                x_paths.extend([os.path.join(img_dir, f) for f in img_files])
                
                if label == 'good':
                    y_paths.extend([None] * len(img_files))
                    labels.extend([0] * len(img_files))  # 0 for normal samples
                else:
                    # For defective samples, also get their mask paths
                    mask_dir = os.path.join(self.root_path, self.category, 'ground_truth', label)
                    y_paths.extend([os.path.join(mask_dir, f.split('.')[0] + '_mask.png') for f in img_files])
                    labels.extend([1] * len(img_files))  # 1 for anomaly samples

        return x_paths, y_paths, labels

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        # Load and process image
        img_path = self.x_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resize, self.resize))
        
        # Random crop during training, center crop during testing
        if self.is_train:
            # Random crop
            start_x = np.random.randint(0, self.resize - self.cropsize)
            start_y = np.random.randint(0, self.resize - self.cropsize)
        else:
            # Center crop
            start_x = (self.resize - self.cropsize) // 2
            start_y = (self.resize - self.cropsize) // 2
            
        img = img[start_y:start_y + self.cropsize, start_x:start_x + self.cropsize]
        img = self.transform_x(img)

        # Load and process mask if it exists
        mask = None
        if self.y_paths[idx] is not None:
            mask = cv2.imread(self.y_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.resize, self.resize))
            mask = mask[start_y:start_y + self.cropsize, start_x:start_x + self.cropsize]
            mask = torch.tensor(mask > 0, dtype=torch.float32)
        else:
            mask = torch.zeros((self.cropsize, self.cropsize), dtype=torch.float32)

        label = self.labels[idx]
        
        return img, mask, label
