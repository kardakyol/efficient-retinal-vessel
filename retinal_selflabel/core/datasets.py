import random
import re
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# discover the datasets and their image mask pairs
def feature_discovery(directory):
    root_file = Path(directory)
    datasets = ["DRIVE", "CHASE", "HRF", "STARE"]
    samples = []
    
    # culumative valid image types
    all_files = []
    valid_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.ppm'}
    
    for file in root_file.rglob("*"):
        if file.is_file() and not file.name.startswith(".") and file.suffix.lower() in valid_extensions:
            all_files.append(file)
            
    # discover HRF dataset with highest-resolution images
    hrf_images, hrf_masks = [], []
    
    for file in all_files:
        if file.parent.name.lower() == "images" and "HRF" in str(file).upper():
            hrf_images.append(file)
        elif file.parent.name.lower() == "manual1" and "HRF" in str(file).upper():
            hrf_masks.append(file)
    
    for image in hrf_images: 
        target_mask = next((m for m in hrf_masks if m.stem == image.stem), None)
        if target_mask:
            samples.append(
                {"image_path": str(image), 
                 "mask_path": str(target_mask), 
                 "dataset": "HRF", 
                 "id": image.stem})
            
    
    # discover Drive dataset
    drive_images, drive_masks = [], []
    
    for file in all_files:
        if file.parent.name.lower() == "images" and "DRIVE" in str(file).upper():
            drive_images.append(file)
        elif file.parent.name.lower() == "1st_manual" and "DRIVE" in str(file).upper():
            drive_masks.append(file)
            
    for image in drive_images:
        image_id = image.stem.split("_")[0]
        target_mask = next((m for m in drive_masks if m.stem.split("_")[0] == image_id), None)
        if target_mask:
            samples.append(
                {"image_path": str(image), 
                 "mask_path": str(target_mask), 
                 "dataset": "DRIVE", 
                 "id": image.stem})
            
    # discover Chase dataset
    chase_images, chase_masks = [], []
    
    for file in all_files:
        if file.parent.name.lower() == "images" and "CHASE" in str(file).upper():
            chase_images.append(file)
        elif file.parent.name.lower() == "masks" and "CHASE" in str(file).upper() and "1stHO" in file.name:
            chase_masks.append(file)
            
    for image in chase_images:
        target_mask = next((m for m in chase_masks if image.stem in m.stem), None)
        if target_mask:
            samples.append(
                {"image_path": str(image), 
                 "mask_path": str(target_mask), 
                 "dataset": "CHASE", 
                 "id": image.stem})
            
    
    # reporting the results of dataset discovery
    print(f"Found {len(samples)} matched to image-mask pairs")
    for dataset in datasets:
        count = 0
        for sample in samples:
            if sample["dataset"] == dataset:
                count += 1
        if count > 0:
            print(f"{dataset}: {count} pairs")
    
    if not samples: 
        raise FileNotFoundError("No valid image mask pairs are not found. Check the dataset structure for your file")
    return samples


# transform images for training and validation
def transform_images(transform_type, img_size=512):
    
    if transform_type == "training":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2(),
        ])
    elif transform_type == "validation":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Invalid image type: {transform_type}")


# retinal vessel dataset
class RetinalVesselDataset(Dataset):
    def __init__(self, samples, transform=None, return_metadata= False):
        self.samples = samples
        self.transform = transform
        self.return_metadata = return_metadata
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        
        read_img = cv2.imread(sample["image_path"])
        if read_img is None: raise FileNotFoundError(f"Unable to read image with reference path: {sample['image_path']}")
        img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)
        
        img_mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        if img_mask is None: raise FileNotFoundError(f"Unable to read mask with reference path: {sample['mask_path']}")
        
        img_mask = (img_mask > 127).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=img, mask=img_mask)
            img = augmented['image']
            img_mask = augmented['mask']

        if not torch.is_tensor(img_mask):
            img_mask = torch.from_numpy(np.asarray(img_mask))
        float_mask = img_mask.float().unsqueeze(0)

        if self.return_metadata:
            return img, float_mask, sample
        return img, float_mask


    # extract raw image and mask without transformation for visualiztion later
    def extract_raw(self, index): 
        sample = self.samples[index]
        img = cv2.cvtColor(cv2.imread(sample["image_path"]), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        return img, mask
    

# sparse annotation simulator
class SparseAnnotationSimulator:
    def __init__(self, samples, patch_size = 128, patches_per_image = 1, min_vessel_fraction = 0.01,
                 max_attempts = 50, seed = 42):
        self.samples = samples
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.min_vessel_fraction = min_vessel_fraction
        self.max_attempts = max_attempts
        self.range = np.random.RandomState(seed)
        
        # pre-compute patch coordinates for reproducibility
        self.patch_info = self._compute_patch_locations()
    
    def _compute_patch_locations(self):
        patch_info = []
        for sample_index, sample in enumerate(self.samples):
            mask = (cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
            height, width = mask.shape
            patch_size = self.patch_size
            
            if height < patch_size or width < patch_size:
                continue
            
            patches = []
            for _ in range(self.patches_per_image):
                found = False
                for attempt in range(self.max_attempts):
                    row = self.range.randint(0, height - patch_size + 1)
                    col = self.range.randint(0, width - patch_size + 1)
                    
                    patch_mask = mask[row:row + patch_size, col:col + patch_size]
                    vessel_frac = patch_mask.mean()
                    
                    if vessel_frac >= self.min_vessel_fraction:
                        patches.append((row, col))
                        found = True
                        break
                
                if not found:
                    row = (height - patch_size) // 2
                    col = (width - patch_size) // 2
                    patches.append((row, col))
            
            patch_info.append({
                "sample_idx": sample_index, "patches": patches, "image_shape": (height, width),
            })
        
        sum_patches = 0
        for p in patch_info:
            sum_patches += len(p["patches"])
        
        print(f"Generated {sum_patches} sparse annotation patches")
        return patch_info
    
    def get_annotation_coverage(self):
        total_annotated, total_area = 0,0
        for info in self.patch_info:
            height, width = info["image_shape"]
            total_area += height * width
            total_annotated += len(info["patches"]) * (self.patch_size ** 2)
            
        if total_area > 0: return total_annotated / total_area
        else: return 0
            
# sparse patch dataset
class SparsePatchDataset(Dataset):
    def __init__(self, samples, patch_info, patch_size = 128, transform=None):
        self.samples = samples
        self.patch_size = patch_size
        self.transform = transform
        
        self.flat_patches = []
        for info in patch_info:
            for (row, col) in info["patches"]:
                self.flat_patches.append({
                    "sample_idx": info["sample_idx"],
                    "row": row,
                    "col": col,
                })
    
    def __len__(self):
        return len(self.flat_patches)
    
    def __getitem__(self, index):
        patch = self.flat_patches[index]
        sample = self.samples[patch["sample_idx"]]
        row, col = patch["row"], patch["col"]
        patch_size = self.patch_size
        
        image = cv2.cvtColor(cv2.imread(sample["image_path"]), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        
        image_patch = image[row:row + patch_size, col:col + patch_size]
        mask_patch = mask[row:row + patch_size, col:col + patch_size]
        
        if self.transform:
            augmented = self.transform(image=image_patch, mask=mask_patch)
            image_patch = augmented["image"]
            mask_patch = augmented["mask"]
        
        mask_patch = mask_patch.float().unsqueeze(0)
        return image_patch, mask_patch

def get_sparse_training_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5), A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]), ToTensorV2(),
    ])

# data splitting by sample
def sample_splitting(samples, test_frac=0.2, val_frac=0.15, seed=42):
    
    range = random.Random(seed)
    per_dataset = {}
    for sample in samples: 
        datasets = sample["dataset"]
        if datasets not in per_dataset:
            per_dataset[datasets] = []
        per_dataset[datasets].append(sample)
        
    train, test = [], []
    for datasets, datasets_samples in per_dataset.items(): 
        range.shuffle(datasets_samples)
        n_test = max(1, int(len(datasets_samples) * test_frac))
        
        test.extend(datasets_samples[:n_test])
        train.extend(datasets_samples[n_test:])
    
    print(f'Split: {len(train)} train, {len(test)} test')
    

    return train, test

# Data splitting by dataset (train, validation, and test)
def dataset_splitting(samples, test_frac=0.2, val_frac=0.15, seed=42):
    train_pool, test = sample_splitting(samples, test_frac=test_frac, seed=seed)
    rel_val_fraction = val_frac / max(1e-9, (1.0 - test_frac))
    rel_val_fraction = min(max((rel_val_fraction), 0.0), 0.9)
    train, val = sample_splitting(train_pool, test_frac=rel_val_fraction, seed=seed + 1)
    print(f'Split: {len(train)} train, {len(val)} val, {len(test)} test')
    return train, val, test


# statistics for dataset
def dataset_statistics(samples): 
    resolutions, vessel_fractions = [], []
    for sample in samples:
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(sample["image_path"])
        if image is not None and mask is not None:
            height, width = image.shape[:2]
            resolutions.append((height, width))
            vessel_frac = (mask > 127).mean()
            vessel_fractions.append(vessel_frac)
    
    if resolutions:
        heights, widths = [], []
        for r in resolutions:
            heights.append(r[0])
            widths.append(r[1])
        print(f'Resolutions \n Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}')
        print(f'Width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}')
    
    if vessel_fractions:
        print(f'Vessel Fraction \n Min: {min(vessel_fractions):.4f}, Max: {max(vessel_fractions):.4f}, Mean: {np.mean(vessel_fractions):.4f}, Std: {np.std(vessel_fractions):.4f}')
    
    datasets = set()
    for sample in samples: 
        dataset = sample["dataset"]
        datasets.add(dataset)
        
    for dataset in sorted(datasets):
        datasets_samples = []
        for sample in samples: 
            if sample["dataset"] == dataset:
                datasets_samples.append(sample)
        print(f'\n {dataset}: {len(datasets_samples)} images')
    
    return resolutions, vessel_fractions