#!/usr/bin/env python3
"""
Utility Functions for Text-to-Sign Model Training
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# RANDOM SEED
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


# =============================================================================
# DEVICE UTILITIES
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    memory = {}
    
    if torch.cuda.is_available():
        memory['cuda_allocated'] = torch.cuda.memory_allocated() / 1e9
        memory['cuda_reserved'] = torch.cuda.memory_reserved() / 1e9
        memory['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / 1e9
    
    import psutil
    process = psutil.Process(os.getpid())
    memory['ram_used'] = process.memory_info().rss / 1e9
    
    return memory


def clear_cuda_cache():
    """Clear CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("CUDA cache cleared")


# =============================================================================
# FILE UTILITIES
# =============================================================================

def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: str):
    """Save dictionary to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_file_size(path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


# =============================================================================
# TIME UTILITIES
# =============================================================================

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_time_str() -> str:
    """Get current time string."""
    return datetime.now().strftime('%H:%M:%S')


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class TrainingLogger:
    """Logger for training progress."""
    
    def __init__(self, log_dir: str = "./logs/"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"training_{get_timestamp()}.log"
        
        # Setup file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        logger.addHandler(file_handler)
    
    def log_config(self, config: Dict):
        """Log configuration."""
        logger.info("=" * 60)
        logger.info("CONFIGURATION")
        logger.info("=" * 60)
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch + 1}/{total_epochs} | {get_time_str()}")
        logger.info(f"{'='*60}")
    
    def log_batch_progress(
        self,
        batch: int,
        total: int,
        loss: float,
        lr: float,
        memory: Optional[Dict] = None,
    ):
        """Log batch progress."""
        msg = f"Batch {batch + 1}/{total} | Loss: {loss:.4f} | LR: {lr:.6f}"
        
        if memory:
            msg += f" | VRAM: {memory.get('cuda_allocated', 0):.2f}GB"
        
        logger.info(msg)
    
    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        duration: float,
    ):
        """Log epoch end summary."""
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Duration: {format_duration(duration)}")


# =============================================================================
# TENSOR UTILITIES
# =============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def move_to_device(data: Any, device: torch.device) -> Any:
    """Move data to device recursively."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    else:
        return data


# =============================================================================
# SKELETON UTILITIES
# =============================================================================

def normalize_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """
    Normalize skeleton coordinates to [0, 1] range.
    
    Args:
        skeleton: [frames, keypoints, coords] skeleton data
    
    Returns:
        Normalized skeleton
    """
    min_vals = skeleton.min(axis=(0, 1), keepdims=True)
    max_vals = skeleton.max(axis=(0, 1), keepdims=True)
    
    normalized = (skeleton - min_vals) / (max_vals - min_vals + 1e-9)
    
    return normalized


def smooth_skeleton(skeleton: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply temporal smoothing to skeleton sequence.
    
    Args:
        skeleton: [frames, keypoints, coords] skeleton data
        window: Smoothing window size
    
    Returns:
        Smoothed skeleton
    """
    from scipy.ndimage import uniform_filter1d
    
    # Apply uniform filter along time axis
    smoothed = uniform_filter1d(skeleton, size=window, axis=0, mode='nearest')
    
    return smoothed


def interpolate_skeleton(
    skeleton: np.ndarray,
    target_frames: int,
) -> np.ndarray:
    """
    Interpolate skeleton sequence to target frame count.
    
    Args:
        skeleton: [frames, keypoints, coords] skeleton data
        target_frames: Target number of frames
    
    Returns:
        Interpolated skeleton
    """
    from scipy.interpolate import interp1d
    
    original_frames = skeleton.shape[0]
    
    if original_frames == target_frames:
        return skeleton
    
    # Create interpolation function for each keypoint and coordinate
    x_old = np.linspace(0, 1, original_frames)
    x_new = np.linspace(0, 1, target_frames)
    
    interpolated = np.zeros((target_frames, skeleton.shape[1], skeleton.shape[2]))
    
    for k in range(skeleton.shape[1]):
        for c in range(skeleton.shape[2]):
            f = interp1d(x_old, skeleton[:, k, c], kind='cubic')
            interpolated[:, k, c] = f(x_new)
    
    return interpolated


# =============================================================================
# AUGMENTATION UTILITIES
# =============================================================================

def augment_skeleton(
    skeleton: np.ndarray,
    rotation_range: float = 15.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    translation_range: float = 0.05,
    noise_std: float = 0.01,
) -> np.ndarray:
    """
    Apply random augmentations to skeleton.
    
    Args:
        skeleton: [frames, keypoints, coords] skeleton data
        rotation_range: Maximum rotation angle in degrees
        scale_range: Scale range (min, max)
        translation_range: Maximum translation
        noise_std: Gaussian noise standard deviation
    
    Returns:
        Augmented skeleton
    """
    augmented = skeleton.copy()
    
    # Random rotation around z-axis
    angle = np.random.uniform(-rotation_range, rotation_range) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    for i in range(augmented.shape[0]):
        augmented[i] = augmented[i] @ rotation_matrix.T
    
    # Random scale
    scale = np.random.uniform(*scale_range)
    augmented *= scale
    
    # Random translation
    translation = np.random.uniform(-translation_range, translation_range, size=(1, 1, 3))
    augmented += translation
    
    # Add noise
    noise = np.random.randn(*augmented.shape) * noise_std
    augmented += noise
    
    # Clip to valid range
    augmented = np.clip(augmented, 0, 1)
    
    return augmented


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def compute_mpjpe(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Mean Per Joint Position Error.
    
    Args:
        pred: Predicted skeleton [frames, keypoints, coords]
        target: Target skeleton [frames, keypoints, coords]
    
    Returns:
        MPJPE in normalized units
    """
    return np.mean(np.sqrt(np.sum((pred - target) ** 2, axis=-1)))


def compute_acceleration_error(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute acceleration error for motion smoothness.
    
    Args:
        pred: Predicted skeleton [frames, keypoints, coords]
        target: Target skeleton [frames, keypoints, coords]
    
    Returns:
        Acceleration error
    """
    pred_accel = np.diff(np.diff(pred, axis=0), axis=0)
    target_accel = np.diff(np.diff(target, axis=0), axis=0)
    
    return np.mean(np.abs(pred_accel - target_accel))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test device
    device = get_device()
    print(f"\nDevice: {device}")
    
    # Test memory
    memory = get_memory_usage()
    print(f"Memory: {memory}")
    
    # Test skeleton utilities
    skeleton = np.random.rand(60, 543, 3)
    
    normalized = normalize_skeleton(skeleton)
    print(f"\nNormalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    smoothed = smooth_skeleton(skeleton)
    print(f"Smoothed shape: {smoothed.shape}")
    
    interpolated = interpolate_skeleton(skeleton, 100)
    print(f"Interpolated shape: {interpolated.shape}")
    
    augmented = augment_skeleton(skeleton)
    print(f"Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    print("\n✅ All utilities tested successfully!")
