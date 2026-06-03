#!/usr/bin/env python3
"""
Memory-Efficient Dataset for Sign Language Training
Optimized for 4GB VRAM + 16GB RAM hardware constraints
Features: Lazy loading, small batch processing, on-demand data loading
"""

import os
import json
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class SmallBatchSignLanguageDataset(Dataset):
    """
    Memory-efficient dataset that loads only small batches.
    Uses lazy loading to minimize memory footprint.
    
    Args:
        class_dirs: Directory containing organized class folders
        videos_per_class: Number of videos to use per class (default: 10)
        max_frames: Maximum frames per video (default: 60)
        use_augmentation: Whether to include augmented videos (default: True)
        transform: Optional transforms to apply
    """
    
    def __init__(
        self,
        class_dirs: str,
        videos_per_class: int = 10,
        max_frames: int = 60,
        use_augmentation: bool = True,
        transform: Optional[Any] = None,
        split: str = "train",
        train_ratio: float = 0.8,
    ):
        self.class_dirs = Path(class_dirs)
        self.videos_per_class = videos_per_class
        self.max_frames = max_frames
        self.use_augmentation = use_augmentation
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        
        # Collect all samples (only metadata, not actual data)
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.gloss_to_idx = {}
        self.idx_to_gloss = {}
        
        self._scan_classes()
        
        print(f"[{split.upper()}] Dataset initialized:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Total classes: {len(self.class_to_idx)}")
        print(f"  Videos per class: {videos_per_class}")
        print(f"  Max frames per video: {max_frames}")
    
    def _scan_classes(self):
        """Scan class directories and collect sample metadata."""
        if not self.class_dirs.exists():
            print(f"Warning: Directory {self.class_dirs} does not exist.")
            print("Creating sample structure for demonstration...")
            self._create_sample_structure()
            return
        
        class_folders = sorted([d for d in self.class_dirs.iterdir() if d.is_dir()])
        
        for class_idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name.replace("class_", "").split("_", 1)[-1]
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            
            # Load metadata
            metadata_path = class_folder / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                gloss = metadata.get("gloss", class_name.upper())
                if gloss not in self.gloss_to_idx:
                    gloss_idx = len(self.gloss_to_idx)
                    self.gloss_to_idx[gloss] = gloss_idx
                    self.idx_to_gloss[gloss_idx] = gloss
                
                text_labels = metadata.get("text_labels", [class_name])
            else:
                gloss = class_name.upper()
                text_labels = [class_name]
            
            # Collect regular videos
            regular_dir = class_folder / "regular"
            if regular_dir.exists():
                regular_videos = sorted(regular_dir.glob("*.npy"))
                
                # Split for train/val/test
                n_videos = len(regular_videos)
                n_train = int(n_videos * self.train_ratio)
                n_val = int(n_videos * 0.1)
                
                if self.split == "train":
                    selected_videos = regular_videos[:n_train]
                elif self.split == "val":
                    selected_videos = regular_videos[n_train:n_train + n_val]
                else:
                    selected_videos = regular_videos[n_train + n_val:]
                
                # Limit to videos_per_class
                selected_videos = selected_videos[:self.videos_per_class]
                
                for video_path in selected_videos:
                    self.samples.append({
                        "video_path": str(video_path),
                        "text_label": text_labels[0] if text_labels else class_name,
                        "gloss": gloss,
                        "class_id": class_idx,
                        "class_name": class_name,
                        "type": "regular",
                    })
            
            # Collect augmented videos
            if self.use_augmentation:
                augmented_dir = class_folder / "augmented"
                if augmented_dir.exists():
                    augmented_videos = sorted(augmented_dir.glob("*.npy"))
                    augmented_videos = augmented_videos[:self.videos_per_class // 2]
                    
                    for video_path in augmented_videos:
                        self.samples.append({
                            "video_path": str(video_path),
                            "text_label": text_labels[0] if text_labels else class_name,
                            "gloss": gloss,
                            "class_id": class_idx,
                            "class_name": class_name,
                            "type": "augmented",
                        })
    
    def _create_sample_structure(self):
        """Create sample structure for demonstration when no data exists."""
        sample_classes = [
            "hello", "thank_you", "please", "sorry", "goodbye",
            "yes", "no", "help", "learn", "understand"
        ]
        
        for class_idx, class_name in enumerate(sample_classes):
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            
            gloss = class_name.upper()
            self.gloss_to_idx[gloss] = class_idx
            self.idx_to_gloss[class_idx] = gloss
            
            # Create synthetic sample entries
            for i in range(min(self.videos_per_class, 5)):
                self.samples.append({
                    "video_path": f"synthetic_{class_name}_{i}.npy",
                    "text_label": class_name.replace("_", " "),
                    "gloss": gloss,
                    "class_id": class_idx,
                    "class_name": class_name,
                    "type": "synthetic",
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load video on-demand (lazy loading).
        
        Returns:
            Dictionary containing:
                - skeleton: [max_frames, 543, 3] tensor
                - text: text label string
                - gloss: gloss string
                - class_id: class index
                - mask: attention mask for valid frames
        """
        sample = self.samples[idx]
        
        # Check if this is a synthetic sample
        if sample["video_path"].startswith("synthetic_"):
            skeleton_data = self._generate_synthetic_skeleton(sample)
        else:
            # Load skeleton data (only when needed)
            try:
                skeleton_data = np.load(sample["video_path"])
            except FileNotFoundError:
                # Generate synthetic if file not found
                skeleton_data = self._generate_synthetic_skeleton(sample)
        
        # Ensure correct shape [frames, keypoints, coords]
        if skeleton_data.ndim == 2:
            # Add coordinate dimension if missing
            skeleton_data = skeleton_data.reshape(skeleton_data.shape[0], -1, 3)
        
        original_frames = skeleton_data.shape[0]
        
        # Pad or truncate to max_frames
        if skeleton_data.shape[0] < self.max_frames:
            # Pad with last frame
            padding = np.zeros((self.max_frames - skeleton_data.shape[0], 
                               skeleton_data.shape[1], 
                               skeleton_data.shape[2]))
            skeleton_data = np.concatenate([skeleton_data, padding], axis=0)
        else:
            # Truncate
            skeleton_data = skeleton_data[:self.max_frames]
            original_frames = self.max_frames
        
        # Create attention mask (1 for valid frames, 0 for padding)
        mask = np.zeros(self.max_frames)
        mask[:original_frames] = 1
        
        # Apply transforms if specified
        if self.transform:
            skeleton_data = self.transform(skeleton_data)
        
        return {
            "skeleton": torch.FloatTensor(skeleton_data),
            "text": sample["text_label"],
            "gloss": sample["gloss"],
            "gloss_idx": self.gloss_to_idx.get(sample["gloss"], 0),
            "class_id": sample["class_id"],
            "mask": torch.FloatTensor(mask),
            "num_frames": original_frames,
        }
    
    def _generate_synthetic_skeleton(self, sample: Dict) -> np.ndarray:
        """Generate synthetic skeleton data for demonstration."""
        # 543 keypoints × 3 coordinates
        num_frames = random.randint(20, self.max_frames)
        num_keypoints = 543
        num_coords = 3
        
        # Generate smooth motion using sine waves
        t = np.linspace(0, 2 * np.pi, num_frames)
        
        # Base position (centered, normalized coordinates)
        skeleton = np.zeros((num_frames, num_keypoints, num_coords))
        
        # Body keypoints (33) - basic standing pose with arm movement
        for i in range(33):
            # Base position
            base_x = 0.5 + 0.1 * np.sin(t + i * 0.1)
            base_y = 0.5 + 0.05 * np.cos(t * 2 + i * 0.05)
            base_z = 0.0 + 0.02 * np.sin(t + i * 0.2)
            
            skeleton[:, i, 0] = base_x
            skeleton[:, i, 1] = base_y
            skeleton[:, i, 2] = base_z
        
        # Hand keypoints (42 total) - finger movement
        for i in range(33, 75):
            base_x = 0.5 + 0.15 * np.sin(t * 1.5 + (i - 33) * 0.1)
            base_y = 0.6 + 0.1 * np.cos(t + (i - 33) * 0.1)
            base_z = 0.1 * np.sin(t * 0.5 + (i - 33) * 0.05)
            
            skeleton[:, i, 0] = base_x
            skeleton[:, i, 1] = base_y
            skeleton[:, i, 2] = base_z
        
        # Face keypoints (468) - minimal movement
        for i in range(75, 543):
            base_x = 0.5 + 0.01 * np.sin(t + (i - 75) * 0.001)
            base_y = 0.3 + 0.01 * np.cos(t + (i - 75) * 0.001)
            base_z = 0.05 * np.sin(t * 0.3 + (i - 75) * 0.0005)
            
            skeleton[:, i, 0] = base_x
            skeleton[:, i, 1] = base_y
            skeleton[:, i, 2] = base_z
        
        return skeleton
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        class_counts = {}
        for sample in self.samples:
            class_id = sample["class_id"]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        total = len(self.samples)
        num_classes = len(self.class_to_idx)
        
        weights = torch.zeros(num_classes)
        for class_id, count in class_counts.items():
            weights[class_id] = total / (num_classes * count)
        
        return weights
    
    def get_gloss_weights(self) -> torch.Tensor:
        """Calculate gloss weights for balanced training."""
        gloss_counts = {}
        for sample in self.samples:
            gloss = sample["gloss"]
            gloss_counts[gloss] = gloss_counts.get(gloss, 0) + 1
        
        total = len(self.samples)
        num_glosses = len(self.gloss_to_idx)
        
        weights = torch.zeros(num_glosses)
        for gloss, count in gloss_counts.items():
            if gloss in self.gloss_to_idx:
                weights[self.gloss_to_idx[gloss]] = total / (num_glosses * count)
        
        return weights


class SignLanguageDataModule:
    """
    Data module for managing train/val/test dataloaders.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        num_workers: int = 2,
        videos_per_class: int = 10,
        max_frames: int = 60,
        use_augmentation: bool = True,
        pin_memory: bool = False,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.videos_per_class = videos_per_class
        self.max_frames = max_frames
        self.use_augmentation = use_augmentation
        self.pin_memory = pin_memory
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup train/val/test datasets."""
        self.train_dataset = SmallBatchSignLanguageDataset(
            class_dirs=self.data_dir,
            videos_per_class=self.videos_per_class,
            max_frames=self.max_frames,
            use_augmentation=self.use_augmentation,
            split="train",
        )
        
        self.val_dataset = SmallBatchSignLanguageDataset(
            class_dirs=self.data_dir,
            videos_per_class=self.videos_per_class,
            max_frames=self.max_frames,
            use_augmentation=False,
            split="val",
        )
        
        self.test_dataset = SmallBatchSignLanguageDataset(
            class_dirs=self.data_dir,
            videos_per_class=self.videos_per_class,
            max_frames=self.max_frames,
            use_augmentation=False,
            split="test",
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class SkeletonTransform:
    """Transforms for skeleton data augmentation."""
    
    @staticmethod
    def random_rotation(skeleton: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """Apply random rotation to skeleton."""
        angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotate around z-axis
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Apply to each frame
        rotated = np.zeros_like(skeleton)
        for i in range(skeleton.shape[0]):
            rotated[i] = skeleton[i] @ rotation_matrix.T
        
        return rotated
    
    @staticmethod
    def random_scale(skeleton: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply random scaling to skeleton."""
        scale = np.random.uniform(*scale_range)
        return skeleton * scale
    
    @staticmethod
    def random_translation(skeleton: np.ndarray, max_shift: float = 0.05) -> np.ndarray:
        """Apply random translation to skeleton."""
        shift = np.random.uniform(-max_shift, max_shift, size=(1, 1, 3))
        return skeleton + shift
    
    @staticmethod
    def time_warp(skeleton: np.ndarray, max_warp: float = 0.1) -> np.ndarray:
        """Apply temporal warping to skeleton sequence."""
        num_frames = skeleton.shape[0]
        warp_factor = np.random.uniform(1 - max_warp, 1 + max_warp)
        new_num_frames = int(num_frames * warp_factor)
        
        # Interpolate to new length then back
        indices = np.linspace(0, num_frames - 1, new_num_frames)
        warped = np.zeros((new_num_frames, skeleton.shape[1], skeleton.shape[2]))
        
        for k in range(skeleton.shape[1]):
            for c in range(skeleton.shape[2]):
                warped[:, k, c] = np.interp(indices, np.arange(num_frames), skeleton[:, k, c])
        
        # Resample back to original length
        indices_back = np.linspace(0, new_num_frames - 1, num_frames)
        result = np.zeros_like(skeleton)
        
        for k in range(skeleton.shape[1]):
            for c in range(skeleton.shape[2]):
                result[:, k, c] = np.interp(indices_back, np.arange(new_num_frames), warped[:, k, c])
        
        return result


def create_sample_metadata(class_name: str, class_id: int) -> Dict:
    """Create sample metadata for a class."""
    return {
        "class_name": class_name,
        "class_id": class_id,
        "text_labels": [class_name.replace("_", " ")],
        "gloss": class_name.upper(),
        "sign_language": "ASL",
        "videos": []
    }


def setup_sample_dataset(base_dir: str, num_classes: int = 20, videos_per_class: int = 10):
    """
    Setup a sample dataset structure for testing.
    
    Args:
        base_dir: Base directory for the dataset
        num_classes: Number of classes to create
        videos_per_class: Number of videos per class
    """
    import os
    
    sample_signs = [
        "hello", "thank_you", "please", "sorry", "goodbye",
        "yes", "no", "help", "learn", "understand",
        "want", "need", "like", "love", "happy",
        "sad", "angry", "hungry", "thirsty", "tired"
    ]
    
    organized_dir = Path(base_dir) / "organized_classes"
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sign in enumerate(sample_signs[:num_classes]):
        class_dir = organized_dir / f"class_{i+1:03d}_{sign}"
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create regular and augmented directories
        (class_dir / "regular").mkdir(exist_ok=True)
        (class_dir / "augmented").mkdir(exist_ok=True)
        
        # Create metadata
        metadata = create_sample_metadata(sign, i + 1)
        
        for j in range(videos_per_class):
            video_info = {
                "filename": f"video_{j+1:03d}.npy",
                "type": "regular",
                "duration_frames": np.random.randint(30, 60),
                "fps": 25,
                "signer_id": f"signer_{j % 5 + 1:02d}",
                "skeleton_format": "mediapipe_holistic",
                "keypoints": {
                    "body": 33,
                    "left_hand": 21,
                    "right_hand": 21,
                    "face": 468
                }
            }
            metadata["videos"].append(video_info)
        
        # Save metadata
        with open(class_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created class: {sign}")
    
    print(f"\nDataset structure created at: {organized_dir}")


if __name__ == "__main__":
    # Test the dataset
    print("Testing SignLanguageDataset...")
    
    # Create sample dataset structure
    setup_sample_dataset("./data", num_classes=10, videos_per_class=5)
    
    # Test dataset loading
    dataset = SmallBatchSignLanguageDataset(
        class_dirs="./data/organized_classes/",
        videos_per_class=5,
        max_frames=60,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Skeleton shape: {sample['skeleton'].shape}")
        print(f"  Text: {sample['text']}")
        print(f"  Gloss: {sample['gloss']}")
        print(f"  Class ID: {sample['class_id']}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Num frames: {sample['num_frames']}")
