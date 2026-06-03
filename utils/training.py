#!/usr/bin/env python3
"""
Complete Training Loop with Mixed Precision
Optimized for 4GB VRAM + 16GB RAM hardware constraints
Features:
- Gradient accumulation for effective larger batch size
- Mixed precision training (FP16) for memory efficiency
- Time-based training control
- Automatic checkpointing
- Early stopping
- Learning rate scheduling
"""

import os
import sys
import time
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import TRAINING_CONFIG, MODEL_CONFIG
from models.text_to_sign_model import TextToSignModel
from data.dataset import SmallBatchSignLanguageDataset
from utils.checkpoint import CheckpointManager, TrainingStateTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./logs/training.log'),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# TRAINER CLASS
# =============================================================================

class TextToSignTrainer:
    """
    Complete trainer for Text-to-Sign model.
    Handles training, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: TextToSignModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: TextToSignModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.01),
            eps=config.get('adam_epsilon', 1e-8),
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.use_amp = config.get('mixed_precision', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 8)
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.get('checkpoint_dir', './checkpoints/'),
            save_interval_minutes=config.get('checkpoint_frequency', 30),
            keep_last_n=config.get('keep_last_n_checkpoints', 5),
            training_start_time=config.get('training_start_time', '22:00'),
            training_end_time=config.get('training_end_time', '05:00'),
        )
        
        # State tracker
        self.state_tracker = TrainingStateTracker(log_dir='./logs/')
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 15)
        
        # Max epochs
        self.max_epochs = config.get('max_epochs', 100)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler_type', 'cosine_annealing_warm_restarts')
        
        if scheduler_type == 'cosine_annealing_warm_restarts':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('scheduler_T_0', 10),
                T_mult=self.config.get('scheduler_T_mult', 2),
            )
        elif scheduler_type == 'one_cycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('learning_rate', 3e-4),
                epochs=self.config.get('max_epochs', 100),
                steps_per_epoch=len(self.train_loader),
            )
        else:
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )
    
    def train(self, resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            resume_from_checkpoint: Whether to resume from last checkpoint
        
        Returns:
            Training summary dictionary
        """
        logger.info("="*60)
        logger.info("STARTING TRAINING")
        logger.info("="*60)
        
        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            self._resume_from_checkpoint()
        
        # Start training state
        self.state_tracker.start_training()
        
        # Training loop
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Check time-based stop
            if not self.checkpoint_manager.should_continue_training():
                logger.info("Time limit reached. Stopping training...")
                self._save_checkpoint("time_limit_reached")
                break
            
            # Train one epoch
            self.state_tracker.start_epoch(epoch)
            train_loss = self._train_one_epoch(epoch)
            
            # Validate
            val_loss = self._validate()
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Log epoch metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.state_tracker.end_epoch(epoch, train_loss, val_loss, current_lr)
            
            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch+1}")
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint("best")
                logger.info(f"✅ New best model! Val Loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training complete
        summary = self.state_tracker.get_training_summary()
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Total epochs: {self.current_epoch + 1}")
        
        return summary
    
    def _train_one_epoch(self, epoch: int) -> float:
        """
        Train one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress tracking
        log_interval = self.config.get('log_every_n_steps', 10)
        
        # Zero gradients at start
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Check for time-based checkpoint
            if self.checkpoint_manager.should_save_checkpoint():
                self._save_checkpoint("time_interval")
            
            # Move batch to device
            skeleton = batch["skeleton"].to(self.device)
            class_ids = batch["class_id"].to(self.device)
            gloss_idxs = batch["gloss_idx"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            # Create dummy input_ids for text (would normally be tokenized text)
            batch_size = skeleton.shape[0]
            input_ids = torch.randint(0, 1000, (batch_size, 20), device=self.device)
            attention_mask = torch.ones(batch_size, 20, device=self.device)
            gloss_ids = gloss_idxs.unsqueeze(1).expand(-1, 10)  # Dummy gloss sequence
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gloss_ids=gloss_ids,
                    target_poses=skeleton,
                    teacher_forcing=True,
                )
                
                loss = outputs.get("losses", {}).get("total_loss", torch.tensor(0.0))
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                vram_usage = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                
                logger.info(
                    f"  Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"VRAM: {vram_usage:.2f}GB"
                )
                
                self.state_tracker.record_vram_usage(vram_usage)
        
        return total_loss / num_batches
    
    def _validate(self) -> float:
        """
        Run validation.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                skeleton = batch["skeleton"].to(self.device)
                class_ids = batch["class_id"].to(self.device)
                gloss_idxs = batch["gloss_idx"].to(self.device)
                
                # Create dummy inputs
                batch_size = skeleton.shape[0]
                input_ids = torch.randint(0, 1000, (batch_size, 20), device=self.device)
                attention_mask = torch.ones(batch_size, 20, device=self.device)
                gloss_ids = gloss_idxs.unsqueeze(1).expand(-1, 10)
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        gloss_ids=gloss_ids,
                        target_poses=skeleton,
                        teacher_forcing=True,
                    )
                    
                    loss = outputs.get("losses", {}).get("total_loss", torch.tensor(0.0))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _save_checkpoint(self, checkpoint_type: str = "regular"):
        """Save training checkpoint."""
        self.checkpoint_manager.current_epoch = self.current_epoch
        self.checkpoint_manager.global_step = self.global_step
        self.checkpoint_manager.best_val_loss = self.best_val_loss
        self.checkpoint_manager.epochs_without_improvement = self.epochs_without_improvement
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            checkpoint_type=checkpoint_type,
        )
    
    def _resume_from_checkpoint(self):
        """Resume training from last checkpoint."""
        checkpoint = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        
        if checkpoint:
            self.current_epoch = checkpoint.get('epoch', 0) + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            
            logger.info(f"Resumed from epoch {self.current_epoch}")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    scaler: Optional[GradScaler] = None,
    epoch: int = 0,
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        config: Training configuration
        scaler: Gradient scaler for mixed precision
        epoch: Current epoch number
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    use_amp = scaler is not None
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 8)
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        skeleton = batch["skeleton"].to(device)
        class_ids = batch["class_id"].to(device)
        gloss_idxs = batch["gloss_idx"].to(device)
        
        # Create dummy inputs
        batch_size = skeleton.shape[0]
        input_ids = torch.randint(0, 1000, (batch_size, 20), device=device)
        attention_mask = torch.ones(batch_size, 20, device=device)
        gloss_ids = gloss_idxs.unsqueeze(1).expand(-1, 10)
        
        # Forward pass
        with autocast(enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gloss_ids=gloss_ids,
                target_poses=skeleton,
                teacher_forcing=True,
            )
            
            loss = outputs.get("losses", {}).get("total_loss", torch.tensor(0.0))
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """
    Validate model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device to validate on
        use_amp: Whether to use mixed precision
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch["skeleton"].to(device)
            class_ids = batch["class_id"].to(device)
            gloss_idxs = batch["gloss_idx"].to(device)
            
            batch_size = skeleton.shape[0]
            input_ids = torch.randint(0, 1000, (batch_size, 20), device=device)
            attention_mask = torch.ones(batch_size, 20, device=device)
            gloss_ids = gloss_idxs.unsqueeze(1).expand(-1, 10)
            
            with autocast(enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gloss_ids=gloss_ids,
                    target_poses=skeleton,
                    teacher_forcing=True,
                )
                
                loss = outputs.get("losses", {}).get("total_loss", torch.tensor(0.0))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main training function."""
    import argparse
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description="Train Text-to-Sign Model")
    parser.add_argument('--data_dir', type=str, default='./data/organized_classes/',
                       help='Path to data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                       help='Path to checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    args = parser.parse_args()
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path('./logs/').mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    
    train_dataset = SmallBatchSignLanguageDataset(
        class_dirs=args.data_dir,
        videos_per_class=10,
        max_frames=60,
        split="train",
    )
    
    val_dataset = SmallBatchSignLanguageDataset(
        class_dirs=args.data_dir,
        videos_per_class=10,
        max_frames=60,
        split="val",
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )
    
    # Create model
    logger.info("Creating model...")
    
    config = {
        'vocab_size': 10000,
        'text_embedding_dim': 768,
        'text_hidden_dim': 768,
        'text_encoder_layers': 4,
        'gloss_vocab_size': 100,
        'gloss_hidden_dim': 512,
        'gloss_encoder_layers': 4,
        'gloss_decoder_layers': 4,
        'gloss_embed_dim': 256,
        'pose_hidden_dim': 512,
        'pose_layers': 6,
        'refine_hidden_dim': 256,
        'refine_layers': 9,
        'pose_dim': 543,
        'pose_coords': 3,
        'num_heads': 8,
        'max_frames': 60,
        'max_text_length': 128,
        'temporal_kernel': 9,
        'dropout': 0.1,
    }
    
    model = TextToSignModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Training config
    training_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': 8,
        'max_epochs': args.max_epochs,
        'early_stopping_patience': 15,
        'checkpoint_dir': args.checkpoint_dir,
        'checkpoint_frequency': 30,
        'keep_last_n_checkpoints': 5,
        'training_start_time': '22:00',
        'training_end_time': '05:00',
        'scheduler_type': 'cosine_annealing_warm_restarts',
        'scheduler_T_0': 10,
        'scheduler_T_mult': 2,
        'log_every_n_steps': 10,
    }
    
    # Create trainer
    trainer = TextToSignTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
    )
    
    # Train
    summary = trainer.train(resume_from_checkpoint=args.resume)
    
    logger.info("\nTraining Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
