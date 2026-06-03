#!/usr/bin/env python3
"""
Inference Script for Text-to-Sign Translation

Usage:
    python inference.py --text "Hello, how are you?" --model_path ./checkpoints/best_model.pth
    python inference.py --interactive  # Interactive mode
    python inference.py --batch input.txt outputs/  # Batch mode
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from configs.config import MODEL_CONFIG, OUTPUT_CONFIG
from utils.inference import TextToSignInference, generate_sign_language_video
from utils.visualization import create_skeleton_video, create_skeleton_gif
from utils.helpers import ensure_dir, get_timestamp


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Text-to-Sign Translation Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input
    parser.add_argument('--text', type=str, default=None,
                       help='Input text to translate')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--batch', nargs=2, metavar=('INPUT_FILE', 'OUTPUT_DIR'),
                       help='Batch mode: process file with one text per line')
    
    # Model
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run inference on')
    
    # Output
    parser.add_argument('--output', type=str, default='./outputs/',
                       help='Output directory or file path')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Output file name (without extension)')
    parser.add_argument('--fps', type=int, default=25,
                       help='Output video FPS')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum frames to generate')
    parser.add_argument('--save_npy', action='store_true', default=True,
                       help='Save skeleton data as NPY file')
    parser.add_argument('--save_gif', action='store_true',
                       help='Also save as GIF')
    
    # Visualization
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080],
                       help='Output resolution (width height)')
    parser.add_argument('--show_face', action='store_true',
                       help='Render face keypoints')
    
    return parser.parse_args()


def translate_single(
    text: str,
    inference_pipeline: TextToSignInference,
    args,
) -> dict:
    """Translate a single text input."""
    logger.info(f"Translating: '{text}'")
    
    # Translate
    result = inference_pipeline.translate(
        text,
        max_frames=args.max_frames,
        return_intermediate=True,
    )
    
    return result


def save_outputs(
    result: dict,
    output_dir: str,
    output_name: str,
    args,
):
    """Save translation outputs."""
    ensure_dir(output_dir)
    
    # Create file paths
    if output_name is None:
        output_name = f"sign_{get_timestamp()}"
    
    npy_path = os.path.join(output_dir, f"{output_name}.npy")
    mp4_path = os.path.join(output_dir, f"{output_name}.mp4")
    gif_path = os.path.join(output_dir, f"{output_name}.gif")
    
    # Save skeleton data
    if args.save_npy:
        np.save(npy_path, result['skeleton'])
        logger.info(f"Skeleton saved: {npy_path}")
    
    # Create video
    create_skeleton_video(
        result['skeleton'],
        mp4_path,
        fps=args.fps,
        resolution=tuple(args.resolution),
        show_face=args.show_face,
    )
    logger.info(f"Video saved: {mp4_path}")
    
    # Create GIF
    if args.save_gif:
        create_skeleton_gif(
            result['skeleton'],
            gif_path,
            fps=args.fps,
            resolution=(480, 270),
        )
        logger.info(f"GIF saved: {gif_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRANSLATION COMPLETE")
    print(f"{'='*60}")
    print(f"Input: {result.get('text', 'N/A')}")
    print(f"Gloss: {result['gloss_text']}")
    print(f"Frames: {result['skeleton'].shape[0]}")
    print(f"Keypoints: {result['skeleton'].shape[1]}")
    print(f"Output: {mp4_path}")
    print(f"{'='*60}\n")


def interactive_mode(inference_pipeline: TextToSignInference, args):
    """Run interactive translation mode."""
    print("\n" + "=" * 60)
    print("TEXT-TO-SIGN INTERACTIVE MODE")
    print("=" * 60)
    print("Enter text to translate (or 'quit' to exit)")
    print("=" * 60 + "\n")
    
    while True:
        try:
            text = input(">>> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            # Translate
            result = translate_single(text, inference_pipeline, args)
            
            # Save outputs
            output_name = text.replace(" ", "_")[:30] + f"_{get_timestamp()}"
            save_outputs(result, args.output, output_name, args)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def batch_mode(input_file: str, output_dir: str, inference_pipeline: TextToSignInference, args):
    """Process batch of texts from file."""
    logger.info(f"Processing batch from: {input_file}")
    
    with open(input_file, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Found {len(texts)} texts to process")
    
    for i, text in enumerate(texts):
        logger.info(f"\nProcessing {i+1}/{len(texts)}: '{text}'")
        
        result = translate_single(text, inference_pipeline, args)
        output_name = f"sign_{i+1:04d}_{text.replace(' ', '_')[:20]}"
        save_outputs(result, output_dir, output_name, args)
    
    logger.info(f"\nBatch processing complete. {len(texts)} videos saved to {output_dir}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("TEXT-TO-SIGN TRANSLATION")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Initialize inference pipeline
    logger.info("Initializing inference pipeline...")
    inference_pipeline = TextToSignInference(
        model_path=args.model_path,
        device=args.device,
    )
    
    # Run in appropriate mode
    if args.interactive:
        interactive_mode(inference_pipeline, args)
    elif args.batch:
        batch_mode(args.batch[0], args.batch[1], inference_pipeline, args)
    elif args.text:
        result = translate_single(args.text, inference_pipeline, args)
        output_name = args.output_name or f"sign_{get_timestamp()}"
        save_outputs(result, args.output, output_name, args)
    else:
        # Default: interactive mode
        interactive_mode(inference_pipeline, args)


if __name__ == "__main__":
    main()
