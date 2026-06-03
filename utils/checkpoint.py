#!/usr/bin/env python3
"""
Checkpoint and Resume System for Night Training
Features:
- Automatic checkpoint saving every N minutes
- Auto-resume from last checkpoint
- Time-based training control (10 PM - 5 AM)
- Graceful shutdown with state preservation
"""

import os
import json
import glob
import random
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """
    Manages model checkpoints with automatic saving and resumption.
    Optimized for night training schedule (10 PM - 5 AM).
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints/",
        save_interval_minutes: int = 30,
        keep_last_n: int = 5,
        training_start_time: str = "22:00",
        training_end_time: str = "05:00",
        shutdown_buffer_minutes: int = 5,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval_minutes: Save checkpoint every N minutes
            keep_last_n: Number of recent checkpoints to keep
            training_start_time: Training window start (HH:MM format)
            training_end_time: Training window end (HH:MM format)
            shutdown_buffer_minutes: Stop N minutes before end time
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = timedelta(minutes=save_interval_minutes)
        self.keep_last_n = keep_last_n
        
        self.training_start_time = training_start_time
        self.training_end_time = training_end_time
        self.shutdown_buffer = timedelta(minutes=shutdown_buffer_minutes)
        
        self.last_save_time = datetime.now()
        self.training_start_dt = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.train_loss_history = []
        self.val_loss_history = []
        
        logger.info(f"CheckpointManager initialized:")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"  Save interval: {save_interval_minutes} minutes")
        logger.info(f"  Training window: {training_start_time} - {training_end_time}")
    
    def should_save_checkpoint(self) -> bool:
        """Check if it's time to save a checkpoint."""
        time_since_last_save = datetime.now() - self.last_save_time
        return time_since_last_save >= self.save_interval
    
    def should_continue_training(self) -> bool:
        """
        Check if current time is within training window.
        Returns False if approaching end time.
        """
        current_time = datetime.now().time()
        
        # Parse end time
        end_hour, end_minute = map(int, self.training_end_time.split(':'))
        end_time = datetime.now().replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        
        # Apply buffer
        end_with_buffer = end_time - self.shutdown_buffer
        
        # Check if we've passed the end time (with buffer)
        if datetime.now() >= end_with_buffer:
            logger.info(f"Approaching end time ({self.training_end_time}). Saving checkpoint and stopping...")
            return False
        
        return True
    
    def get_time_until_end(self) -> timedelta:
        """Get remaining time until training end."""
        end_hour, end_minute = map(int, self.training_end_time.split(':'))
        end_time = datetime.now().replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        
        # Handle overnight training (e.g., 22:00 to 05:00)
        if end_time < datetime.now():
            end_time += timedelta(days=1)
        
        return end_time - datetime.now()
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        checkpoint_type: str = "regular",
        extra_info: Optional[Dict] = None,
    ) -> str:
        """
        Save a comprehensive checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer
            scheduler: Learning rate scheduler (optional)
            scaler: Gradient scaler for mixed precision (optional)
            checkpoint_type: Type of checkpoint (regular, best, epoch_N, etc.)
            extra_info: Additional information to save
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if checkpoint_type == "regular":
            filename = f"checkpoint_{timestamp}.pth"
        elif checkpoint_type == "best":
            filename = "best_model.pth"
        elif checkpoint_type.startswith("epoch_"):
            filename = f"checkpoint_{checkpoint_type}.pth"
        else:
            filename = f"checkpoint_{checkpoint_type}_{timestamp}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Build checkpoint dictionary
        checkpoint = {
            # Model state
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            # Training progress
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            
            # Loss history
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            
            # Time tracking
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_start_time': self.training_start_dt,
            'last_checkpoint_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # Random states for reproducibility
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        
        # Add scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add scaler state
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # Add extra info
        if extra_info is not None:
            checkpoint['extra_info'] = extra_info
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update last save time
        self.last_save_time = datetime.now()
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file (if None, loads latest)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            scaler: Gradient scaler to load state into
            load_best: If True, load best_model.pth instead of latest
        
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            # Find the latest checkpoint
            if load_best:
                checkpoint_path = str(self.checkpoint_dir / "best_model.pth")
            else:
                checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pth"))
                if len(checkpoints) == 0:
                    logger.warning("No checkpoint found. Starting fresh training...")
                    return {}
                checkpoint_path = str(checkpoints[-1])
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return {}
        
        logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        if model is not None and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("  ✓ Model state loaded")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("  ✓ Optimizer state loaded")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("  ✓ Scheduler state loaded")
        
        # Load scaler state
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("  ✓ Scaler state loaded")
        
        # Restore random states
        if 'random_state' in checkpoint:
            random.setstate(checkpoint['random_state'])
        if 'numpy_random_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_random_state'])
        if 'torch_random_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_random_state'])
        if checkpoint.get('cuda_random_state') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        
        logger.info(f"  ✓ Resumed from epoch {self.current_epoch}")
        logger.info(f"  ✓ Best validation loss: {self.best_val_loss:.4f}")
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pth"))
        
        # Don't delete best_model.pth
        checkpoints = [c for c in checkpoints if "best" not in c.name]
        
        # Keep only the last N checkpoints
        while len(checkpoints) > self.keep_last_n:
            old_checkpoint = checkpoints.pop(0)
            old_checkpoint.unlink()
            logger.info(f"🗑️  Removed old checkpoint: {old_checkpoint}")
    
    def save_training_resume_info(self, extra_info: Optional[Dict] = None):
        """Save a lightweight JSON file with training resume info."""
        resume_info = {
            'last_checkpoint_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
        }
        
        if extra_info:
            resume_info.update(extra_info)
        
        resume_path = self.checkpoint_dir / "training_resume.json"
        with open(resume_path, 'w') as f:
            json.dump(resume_info, f, indent=2)
    
    def load_training_resume_info(self) -> Dict[str, Any]:
        """Load training resume info from JSON file."""
        resume_path = self.checkpoint_dir / "training_resume.json"
        
        if resume_path.exists():
            with open(resume_path, 'r') as f:
                return json.load(f)
        return {}


# =============================================================================
# TIME-BASED TRAINING CONTROLLER
# =============================================================================

class TrainingTimeController:
    """
    Controls training schedule with time-based start/stop.
    Designed for overnight training (10 PM - 5 AM).
    """
    
    def __init__(
        self,
        start_time: str = "22:00",
        end_time: str = "05:00",
        shutdown_buffer_minutes: int = 5,
    ):
        """
        Initialize training time controller.
        
        Args:
            start_time: Training window start (HH:MM format)
            end_time: Training window end (HH:MM format)
            shutdown_buffer_minutes: Stop N minutes before end time
        """
        self.start_time = start_time
        self.end_time = end_time
        self.shutdown_buffer = timedelta(minutes=shutdown_buffer_minutes)
        
        self.training_started = False
        self.training_start_dt = None
        
        logger.info(f"TrainingTimeController initialized:")
        logger.info(f"  Training window: {start_time} - {end_time}")
        logger.info(f"  Shutdown buffer: {shutdown_buffer_minutes} minutes")
    
    def is_within_training_window(self) -> bool:
        """Check if current time is within training window."""
        current_time = datetime.now().time()
        
        start_hour, start_minute = map(int, self.start_time.split(':'))
        end_hour, end_minute = map(int, self.end_time.split(':'))
        
        start_time_obj = datetime.now().replace(hour=start_hour, minute=start_minute).time()
        end_time_obj = datetime.now().replace(hour=end_hour, minute=end_minute).time()
        
        # Handle overnight window (e.g., 22:00 - 05:00)
        if start_hour > end_hour:
            # Training crosses midnight
            return current_time >= start_time_obj or current_time < end_time_obj
        else:
            # Training within same day
            return start_time_obj <= current_time < end_time_obj
    
    def should_start_training(self) -> bool:
        """Check if training should start."""
        if self.training_started:
            return False
        
        if self.is_within_training_window():
            self.training_started = True
            self.training_start_dt = datetime.now()
            return True
        
        return False
    
    def should_stop_training(self) -> bool:
        """Check if training should stop (approaching end time)."""
        end_hour, end_minute = map(int, self.end_time.split(':'))
        end_time_dt = datetime.now().replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        
        # Handle overnight
        if end_time_dt < datetime.now():
            end_time_dt += timedelta(days=1)
        
        # Apply buffer
        stop_time = end_time_dt - self.shutdown_buffer
        
        return datetime.now() >= stop_time
    
    def get_remaining_time(self) -> timedelta:
        """Get remaining time in training window."""
        end_hour, end_minute = map(int, self.end_time.split(':'))
        end_time_dt = datetime.now().replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        
        # Handle overnight
        if end_time_dt < datetime.now():
            end_time_dt += timedelta(days=1)
        
        # Apply buffer
        stop_time = end_time_dt - self.shutdown_buffer
        
        remaining = stop_time - datetime.now()
        return remaining if remaining > timedelta(0) else timedelta(0)
    
    def wait_for_training_window(self, check_interval_seconds: int = 60):
        """Wait until training window starts."""
        logger.info(f"Waiting for training window to start at {self.start_time}...")
        
        while not self.should_start_training():
            current_time = datetime.now().strftime('%H:%M:%S')
            logger.info(f"  Current time: {current_time} | Training starts at {self.start_time}")
            
            import time
            time.sleep(check_interval_seconds)
        
        logger.info(f"Training window started at {datetime.now().strftime('%H:%M:%S')}!")


# =============================================================================
# TRAINING STATE TRACKER
# =============================================================================

class TrainingStateTracker:
    """Tracks and logs training state and metrics."""
    
    def __init__(self, log_dir: str = "./logs/"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': [],
            'vram_usage': [],
        }
        
        self.start_time = None
        self.epoch_start_time = None
    
    def start_training(self):
        """Mark training start time."""
        self.start_time = datetime.now()
        logger.info(f"Training started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def start_epoch(self, epoch: int):
        """Mark epoch start time."""
        self.epoch_start_time = datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch} started at {self.epoch_start_time.strftime('%H:%M:%S')}")
        logger.info(f"{'='*60}")
    
    def end_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
    ):
        """Record epoch metrics."""
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['epoch_time'].append(epoch_time)
        
        # Log to file
        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'epoch_time_seconds': epoch_time,
        }
        
        log_file = self.log_dir / f"training_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Print summary
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Learning Rate: {learning_rate:.6f}")
        logger.info(f"  Epoch Time: {epoch_time:.1f}s")
    
    def record_vram_usage(self, usage_gb: float):
        """Record VRAM usage."""
        self.metrics['vram_usage'].append(usage_gb)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if self.start_time is None:
            return {}
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'total_training_time_seconds': total_time,
            'total_epochs': len(self.metrics['train_loss']),
            'best_val_loss': min(self.metrics['val_loss']) if self.metrics['val_loss'] else None,
            'final_train_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
            'final_val_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None,
            'avg_epoch_time': np.mean(self.metrics['epoch_time']) if self.metrics['epoch_time'] else None,
            'max_vram_usage_gb': max(self.metrics['vram_usage']) if self.metrics['vram_usage'] else None,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str = "./checkpoints/",
    checkpoint_type: str = "regular",
):
    """Simple checkpoint save function."""
    manager = CheckpointManager(checkpoint_dir)
    manager.current_epoch = epoch
    manager.train_loss_history.append(loss)
    return manager.save_checkpoint(model, optimizer, checkpoint_type=checkpoint_type)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """Simple checkpoint load function."""
    manager = CheckpointManager()
    return manager.load_checkpoint(checkpoint_path, model, optimizer)


def auto_resume_training(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Tuple[int, float]:
    """
    Auto-resume training from last checkpoint.
    
    Returns:
        Tuple of (start_epoch, best_val_loss)
    """
    manager = CheckpointManager(checkpoint_dir)
    checkpoint = manager.load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    
    if not checkpoint:
        return 0, float('inf')
    
    return manager.current_epoch + 1, manager.best_val_loss


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import torch.nn as nn
    
    print("Testing CheckpointManager...")
    
    # Create a simple model for testing
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Initialize checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir="./test_checkpoints/",
        save_interval_minutes=1,  # Short interval for testing
        keep_last_n=3,
    )
    
    # Test saving
    manager.current_epoch = 5
    manager.global_step = 100
    manager.train_loss_history = [0.5, 0.4, 0.3, 0.25, 0.2]
    
    checkpoint_path = manager.save_checkpoint(
        model, optimizer,
        checkpoint_type="epoch_5"
    )
    
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    
    # Test loading
    model2 = nn.Linear(10, 10)
    optimizer2 = torch.optim.Adam(model2.parameters())
    
    checkpoint = manager.load_checkpoint(
        checkpoint_path,
        model=model2,
        optimizer=optimizer2,
    )
    
    print(f"\nLoaded checkpoint:")
    print(f"  Epoch: {manager.current_epoch}")
    print(f"  Step: {manager.global_step}")
    print(f"  Train loss history: {manager.train_loss_history}")
    
    # Test time controller
    print("\n" + "="*60)
    print("Testing TrainingTimeController...")
    
    time_controller = TrainingTimeController(
        start_time="22:00",
        end_time="05:00",
    )
    
    print(f"Is within training window: {time_controller.is_within_training_window()}")
    print(f"Should stop training: {time_controller.should_stop_training()}")
    print(f"Remaining time: {time_controller.get_remaining_time()}")
    
    print("\n✅ Checkpoint system test passed!")
