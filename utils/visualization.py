#!/usr/bin/env python3
"""
Skeleton Video Visualization Module
Creates animated videos from skeleton pose data

Features:
- OpenCV-based skeleton rendering
- MediaPipe skeleton structure support
- MP4 video output
- Optional GIF export
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not installed. Video generation will be limited.")

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not installed. Animation features will be limited.")


# =============================================================================
# MEDIAPIPE SKELETON CONNECTIONS
# =============================================================================

# Body pose connections (MediaPipe format)
BODY_CONNECTIONS = [
    # Face outline
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # Eyes
    (9, 10),
    # Mouth
    # Left arm
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32),
]

# Hand connections (MediaPipe format)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),            # Palm
]

# Face mesh connections (simplified - just contours)
FACE_CONNECTIONS = [
    # Face oval (simplified)
    (10, 338), (338, 297), (297, 332), (332, 284),
    (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10),
]


# =============================================================================
# SKELETON RENDERER
# =============================================================================

class SkeletonRenderer:
    """
    Renders skeleton poses to images and videos.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        body_color: Tuple[int, int, int] = (0, 255, 0),
        left_hand_color: Tuple[int, int, int] = (255, 0, 0),
        right_hand_color: Tuple[int, int, int] = (0, 0, 255),
        face_color: Tuple[int, int, int] = (255, 255, 255),
        background_color: Tuple[int, int, int] = (0, 0, 0),
        line_thickness: int = 2,
        keypoint_radius: int = 3,
        show_face: bool = False,
    ):
        """
        Initialize skeleton renderer.
        
        Args:
            resolution: Output resolution (width, height)
            body_color: BGR color for body skeleton
            left_hand_color: BGR color for left hand
            right_hand_color: BGR color for right hand
            face_color: BGR color for face
            background_color: BGR color for background
            line_thickness: Line thickness for connections
            keypoint_radius: Radius of keypoint circles
            show_face: Whether to render face keypoints
        """
        self.resolution = resolution
        self.body_color = body_color
        self.left_hand_color = left_hand_color
        self.right_hand_color = right_hand_color
        self.face_color = face_color
        self.background_color = background_color
        self.line_thickness = line_thickness
        self.keypoint_radius = keypoint_radius
        self.show_face = show_face
        
        # Skeleton structure indices
        self.body_start = 0
        self.body_end = 33
        self.left_hand_start = 33
        self.left_hand_end = 54
        self.right_hand_start = 54
        self.right_hand_end = 75
        self.face_start = 75
        self.face_end = 543
    
    def render_frame(
        self,
        skeleton: np.ndarray,
        return_bgr: bool = True,
    ) -> np.ndarray:
        """
        Render a single skeleton frame.
        
        Args:
            skeleton: Skeleton data [543, 3] (keypoints, coords)
            return_bgr: Return BGR format (True) or RGB (False)
        
        Returns:
            Rendered image array
        """
        if not HAS_CV2:
            raise ImportError("OpenCV is required for rendering")
        
        # Create blank image
        img = np.full(
            (self.resolution[1], self.resolution[0], 3),
            self.background_color,
            dtype=np.uint8
        )
        
        # Helper function to convert normalized coords to pixels
        def to_pixel(point: np.ndarray) -> Tuple[int, int]:
            x = int(point[0] * self.resolution[0])
            y = int(point[1] * self.resolution[1])
            return (x, y)
        
        # Draw body
        body_points = skeleton[self.body_start:self.body_end]
        self._draw_connections(
            img, body_points, BODY_CONNECTIONS, self.body_color, to_pixel
        )
        self._draw_keypoints(
            img, body_points, self.body_color, to_pixel
        )
        
        # Draw left hand
        left_hand = skeleton[self.left_hand_start:self.left_hand_end]
        self._draw_connections(
            img, left_hand, HAND_CONNECTIONS, self.left_hand_color, to_pixel
        )
        self._draw_keypoints(
            img, left_hand, self.left_hand_color, to_pixel
        )
        
        # Draw right hand
        right_hand = skeleton[self.right_hand_start:self.right_hand_end]
        self._draw_connections(
            img, right_hand, HAND_CONNECTIONS, self.right_hand_color, to_pixel
        )
        self._draw_keypoints(
            img, right_hand, self.right_hand_color, to_pixel
        )
        
        # Draw face (optional)
        if self.show_face:
            face = skeleton[self.face_start:self.face_end]
            # Draw simplified face (just eyes and mouth region)
            self._draw_keypoints(
                img, face[::10], self.face_color, to_pixel, radius=1  # Subsample
            )
        
        if not return_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def _draw_connections(
        self,
        img: np.ndarray,
        points: np.ndarray,
        connections: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        to_pixel_func,
    ):
        """Draw skeleton connections."""
        for i, j in connections:
            if i < len(points) and j < len(points):
                pt1 = to_pixel_func(points[i])
                pt2 = to_pixel_func(points[j])
                cv2.line(img, pt1, pt2, color, self.line_thickness)
    
    def _draw_keypoints(
        self,
        img: np.ndarray,
        points: np.ndarray,
        color: Tuple[int, int, int],
        to_pixel_func,
        radius: int = None,
    ):
        """Draw skeleton keypoints."""
        radius = radius or self.keypoint_radius
        for point in points:
            pt = to_pixel_func(point)
            cv2.circle(img, pt, radius, color, -1)
    
    def render_video(
        self,
        skeleton_sequence: np.ndarray,
        output_path: str,
        fps: int = 25,
    ) -> str:
        """
        Render skeleton sequence to video.
        
        Args:
            skeleton_sequence: Skeleton data [frames, 543, 3]
            output_path: Output video file path
            fps: Frames per second
        
        Returns:
            Path to saved video
        """
        if not HAS_CV2:
            raise ImportError("OpenCV is required for video generation")
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, self.resolution
        )
        
        num_frames = skeleton_sequence.shape[0]
        logger.info(f"Rendering {num_frames} frames to {output_path}")
        
        for frame_idx in range(num_frames):
            skeleton = skeleton_sequence[frame_idx]
            frame = self.render_frame(skeleton)
            video_writer.write(frame)
            
            if (frame_idx + 1) % 10 == 0:
                logger.info(f"  Rendered {frame_idx + 1}/{num_frames} frames")
        
        video_writer.release()
        logger.info(f"✅ Video saved to: {output_path}")
        
        return output_path


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_skeleton_video(
    skeleton_data: np.ndarray,
    output_path: str,
    fps: int = 25,
    resolution: Tuple[int, int] = (1920, 1080),
    show_face: bool = False,
) -> str:
    """
    Create MP4 video from skeleton data.
    
    Args:
        skeleton_data: [frames, keypoints, coords] numpy array
        output_path: Output video file path
        fps: Frames per second
        resolution: Output resolution (width, height)
        show_face: Whether to render face keypoints
    
    Returns:
        Path to generated video
    """
    renderer = SkeletonRenderer(
        resolution=resolution,
        show_face=show_face,
    )
    
    return renderer.render_video(skeleton_data, output_path, fps)


def create_skeleton_gif(
    skeleton_data: np.ndarray,
    output_path: str,
    fps: int = 25,
    resolution: Tuple[int, int] = (480, 270),
) -> str:
    """
    Create GIF from skeleton data using matplotlib.
    
    Args:
        skeleton_data: [frames, keypoints, coords] numpy array
        output_path: Output GIF file path
        fps: Frames per second
        resolution: Output resolution (width, height)
    
    Returns:
        Path to generated GIF
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for GIF generation")
    
    fig, ax = plt.subplots(figsize=(resolution[0]/100, resolution[1]/100))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.axis('off')
    
    # Initialize plot elements
    body_lines = []
    for _ in BODY_CONNECTIONS:
        line, = ax.plot([], [], 'g-', linewidth=2)
        body_lines.append(line)
    
    body_points, = ax.plot([], [], 'go', markersize=3)
    
    left_hand_points, = ax.plot([], [], 'bo', markersize=2)
    right_hand_points, = ax.plot([], [], 'ro', markersize=2)
    
    def init():
        for line in body_lines:
            line.set_data([], [])
        body_points.set_data([], [])
        left_hand_points.set_data([], [])
        right_hand_points.set_data([], [])
        return body_lines + [body_points, left_hand_points, right_hand_points]
    
    def animate(frame_idx):
        skeleton = skeleton_data[frame_idx]
        
        # Body
        body_x = skeleton[:33, 0]
        body_y = 1 - skeleton[:33, 1]  # Flip y-axis
        body_points.set_data(body_x, body_y)
        
        for i, (conn_i, conn_j) in enumerate(BODY_CONNECTIONS):
            if i < len(body_lines):
                body_lines[i].set_data(
                    [body_x[conn_i], body_x[conn_j]],
                    [body_y[conn_i], body_y[conn_j]]
                )
        
        # Hands
        left_hand = skeleton[33:54]
        left_hand_points.set_data(left_hand[:, 0], 1 - left_hand[:, 1])
        
        right_hand = skeleton[54:75]
        right_hand_points.set_data(right_hand[:, 0], 1 - right_hand[:, 1])
        
        return body_lines + [body_points, left_hand_points, right_hand_points]
    
    anim = FuncAnimation(
        fig, animate,
        init_func=init,
        frames=len(skeleton_data),
        interval=1000/fps,
        blit=True,
    )
    
    # Save GIF
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    
    logger.info(f"✅ GIF saved to: {output_path}")
    
    return output_path


def visualize_skeleton_frame(
    skeleton: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "",
    figsize: Tuple[int, int] = (10, 10),
) -> Optional[np.ndarray]:
    """
    Visualize a single skeleton frame using matplotlib.
    
    Args:
        skeleton: [543, 3] skeleton data
        output_path: Path to save image (optional)
        title: Plot title
        figsize: Figure size
    
    Returns:
        Image array if output_path is None
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for visualization")
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_title(title, color='white')
    ax.axis('off')
    
    # Body
    body_x = skeleton[:33, 0]
    body_y = 1 - skeleton[:33, 1]
    
    for i, j in BODY_CONNECTIONS:
        ax.plot([body_x[i], body_x[j]], [body_y[i], body_y[j]], 'g-', linewidth=2)
    
    ax.plot(body_x, body_y, 'go', markersize=5)
    
    # Left hand
    left_hand = skeleton[33:54]
    ax.plot(left_hand[:, 0], 1 - left_hand[:, 1], 'bo', markersize=3)
    
    # Right hand
    right_hand = skeleton[54:75]
    ax.plot(right_hand[:, 0], 1 - right_hand[:, 1], 'ro', markersize=3)
    
    if output_path:
        plt.savefig(output_path, facecolor='black', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Frame saved to: {output_path}")
        return None
    else:
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Skeleton Video Visualization")
    parser.add_argument('--input', type=str, required=True,
                       help='Input skeleton NPY file')
    parser.add_argument('--output', type=str, default='./outputs/output.mp4',
                       help='Output video file')
    parser.add_argument('--fps', type=int, default=25,
                       help='Frames per second')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080],
                       help='Output resolution (width height)')
    args = parser.parse_args()
    
    # Load skeleton data
    skeleton_data = np.load(args.input)
    print(f"Loaded skeleton data: {skeleton_data.shape}")
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Create video
    create_skeleton_video(
        skeleton_data,
        args.output,
        fps=args.fps,
        resolution=tuple(args.resolution),
    )
    
    print(f"\n✅ Video created: {args.output}")
