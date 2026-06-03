#!/usr/bin/env python3
"""
Main Training Script for Text-to-Sign Translation Model

Usage:
    python train.py --data_dir ./data/organized_classes/ --epochs 100
    python train.py --resume  # Resume from last checkpoint
    python train.py --help    # Show all options

Features:
    - Automatic checkpoint saving every 30 minutes
    - Auto-resume from last checkpoint
    - Time-based training (10 PM - 5 AM)
    - Mixed precision training (FP16)
    - Gradient accumulation for effective larger batch size
    - Early stopping
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from configs.config import TRAINING_CONFIG, MODEL_CONFIG, DATA_CONFIG
from models.text_to_sign_model import TextToSignModel
from data.dataset import SmallBatchSignLanguageDataset, setup_sample_dataset
from utils.checkpoint import CheckpointManager, TrainingStateTracker
from utils.training import TextToSignTrainer
from utils.helpers import (
    set_seed, get_device, ensure_dir, get_timestamp,
    count_parameters, get_model_size, TrainingLogger
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'./logs/training_{get_timestamp()}.log'),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Text-to-Sign Translation Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/organized_classes/',
                       help='Path to organized class data directory')
    parser.add_argument('--videos_per_class', type=int, default=10,
                       help='Number of videos to use per class')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum frames per video')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Model hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (small due to VRAM limit)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gradient_accumulation', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--early_stopping', type=int, default=15,
                       help='Early stopping patience')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                       help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=30,
                       help='Checkpoint frequency in minutes')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Specific checkpoint to resume from')
    
    # Training schedule
    parser.add_argument('--start_time', type=str, default='22:00',
                       help='Training start time (HH:MM)')
    parser.add_argument('--end_time', type=str, default='05:00',
                       help='Training end time (HH:MM)')
    parser.add_argument('--wait_for_start', action='store_true',
                       help='Wait until start time before training')
    
    # Optimization
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision (FP16) training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/',
                       help='Output directory')
    parser.add_argument('--log_dir', type=str, default='./logs/',
                       help='Log directory')
    
    # Setup
    parser.add_argument('--setup_sample_data', action='store_true',
                       help='Setup sample data structure')
    parser.add_argument('--num_sample_classes', type=int, default=20,
                       help='Number of sample classes to create')
    
    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories."""
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.output_dir)
    ensure_dir(args.log_dir)
    ensure_dir('./logs/')
    logger.info("Directories created")


def setup_sample_data(args):
    """Setup sample data for testing."""
    logger.info("Setting up sample data...")
    setup_sample_dataset(
        base_dir='./data/',
        num_classes=args.num_sample_classes,
        videos_per_class=args.videos_per_class,
    )
    logger.info("Sample data setup complete")


def create_model(args):
    """Create the Text-to-Sign model."""
    config = {
        'vocab_size': 10000,
        'text_embedding_dim': 768,
        'text_hidden_dim': 768,
        'text_encoder_layers': 4,
        'gloss_vocab_size': 100,
        'gloss_hidden_dim': args.hidden_dim,
        'gloss_encoder_layers': 4,
        'gloss_decoder_layers': 4,
        'gloss_embed_dim': 256,
        'pose_hidden_dim': args.hidden_dim,
        'pose_layers': args.num_layers,
        'refine_hidden_dim': 256,
        'refine_layers': 9,
        'pose_dim': 543,
        'pose_coords': 3,
        'num_heads': args.num_heads,
        'max_frames': args.max_frames,
        'max_text_length': 128,
        'temporal_kernel': 9,
        'dropout': 0.1,
    }
    
    model = TextToSignModel(config)
    
    # Log model info
    total_params = count_parameters(model)
    model_size = get_model_size(model)
    
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Model size: {model_size:.2f} MB")
    
    return model, config


def create_dataloaders(args):
    """Create data loaders."""
    logger.info("Creating data loaders...")
    
    # Training dataset
    train_dataset = SmallBatchSignLanguageDataset(
        class_dirs=args.data_dir,
        videos_per_class=args.videos_per_class,
        max_frames=args.max_frames,
        use_augmentation=True,
        split="train",
    )
    
    # Validation dataset
    val_dataset = SmallBatchSignLanguageDataset(
        class_dirs=args.data_dir,
        videos_per_class=args.videos_per_class,
        max_frames=args.max_frames,
        use_augmentation=False,
        split="val",
    )
    
    # Data loaders
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
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("TEXT-TO-SIGN TRANSLATION MODEL TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Setup
    setup_directories(args)
    
    # Setup sample data if requested
    if args.setup_sample_data:
        setup_sample_data(args)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create model
    model, model_config = create_model(args)
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(args)
    
    # Training configuration
    training_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': args.gradient_accumulation,
        'max_epochs': args.epochs,
        'early_stopping_patience': args.early_stopping,
        'checkpoint_dir': args.checkpoint_dir,
        'checkpoint_frequency': args.checkpoint_freq,
        'keep_last_n_checkpoints': 5,
        'training_start_time': args.start_time,
        'training_end_time': args.end_time,
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
    logger.info("\nStarting training...")
    summary = trainer.train(resume_from_checkpoint=args.resume)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
