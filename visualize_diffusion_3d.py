#!/usr/bin/env python3
"""
MDAE 3D Dual Corruption Grid Visualization Script

This script generates high-quality 3D volume renderings showing both corruption
types in MDAE training: diffusion noise and spatial masking, arranged in an
L-shaped grid layout.

The visualization shows:
- Top row: Diffusion progression (t=0 to t=T)
- Left column: Masking progression (0% to 95%)

Usage:
    # Generate grid visualization with default settings
    python visualize_diffusion_3d.py

    # Use specific dataset and modality
    python visualize_diffusion_3d.py --dataset brats18 --modality t1ce

    # Custom timesteps
    python visualize_diffusion_3d.py --timesteps 0.0 0.3 0.6 0.9 1.0

    # High-resolution output
    python visualize_diffusion_3d.py --dpi 600

Author: Based on NoiseConditionedMDAETrainer implementation
Date: November 2025
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our utilities
from data_loader_utils import load_brats_sample, create_synthetic_brain_volume
from pyvista_utils import create_dual_corruption_grid_3d


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D visualization of diffusion/denoising process"
    )
    parser.add_argument(
        '--use-synthetic',
        action='store_true',
        help='Use synthetic data instead of real MRI'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='brats18',
        choices=['brats18', 'bratsped'],
        help='BraTS dataset to use (default: brats18)'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='t1ce',
        choices=['t1', 't1ce', 't2', 'flair'],
        help='MRI modality (default: t1ce)'
    )
    parser.add_argument(
        '--case-id',
        type=str,
        default=None,
        help='Specific case ID to load'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='sample_data',
        help='Cache directory (default: sample_data)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output images (default: 300)'
    )
    parser.add_argument(
        '--timesteps',
        type=float,
        nargs='+',
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        help='Timesteps to visualize (default: 0.0 0.1 0.25 0.5 0.75 1.0)'
    )
    parser.add_argument(
        '--sde',
        type=str,
        default='ddpm',
        choices=['ddpm', 've', 'flow'],
        help='SDE type for noise corruption (default: ddpm)'
    )
    parser.add_argument(
        '--transparency',
        type=str,
        default='medium',
        choices=['low', 'medium', 'high'],
        help='Transparency level for volume rendering (default: medium)'
    )

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print("MDAE 3D Dual Corruption Grid Visualization")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load volume
    print("\n1. Loading volume...")
    if not args.use_synthetic:
        try:
            volume, metadata = load_brats_sample(
                case_id=args.case_id,
                dataset=args.dataset,
                modality=args.modality,
                cache_dir=str(cache_dir),
                target_shape=(128, 128, 128)  # Consistent size for visualization
            )
            print(f"   ✓ Loaded BraTS {args.dataset.upper()} sample")
            print(f"   Case ID: {metadata['case_id']}")
            print(f"   Modality: {metadata['modality'].upper()}")
            print(f"   Shape: {volume.shape}")
        except Exception as e:
            print(f"   ✗ Failed to load BraTS data: {e}")
            print(f"   Falling back to synthetic data...")
            args.use_synthetic = True

    if args.use_synthetic:
        volume = create_synthetic_brain_volume(shape=(128, 128, 128), seed=42)
        print(f"   ✓ Created synthetic brain volume")
        print(f"   Shape: {volume.shape}")

    # Generate dual corruption grid visualization
    print(f"\n2. Generating dual corruption grid visualization...")
    print(f"   Diffusion timesteps: {args.timesteps}")
    print(f"   Masking ratios: [0%, 20%, 40%, 60%, 80%, 95%]")
    print(f"   SDE type: {args.sde}")
    print(f"   Transparency: {args.transparency}")

    # Define save paths
    grid_save_path = output_dir / "dual_corruption_grid_3d.png"
    center_save_path = output_dir / "dual_corrupted_center_volume.png"

    plotter = create_dual_corruption_grid_3d(
        volume,
        diffusion_timesteps=args.timesteps,
        masking_ratios=[0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
        patch_size=16,
        transparency_level=args.transparency,
        sde=args.sde,
        save_path=str(grid_save_path),
        save_individual_center=True,
        center_save_path=str(center_save_path)
    )

    # Note: save_publication_image is called inside create_dual_corruption_grid_3d
    plotter.close()

    # Summary
    print("\n" + "=" * 80)
    print("✓ Dual corruption grid visualization generated successfully!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  1. Grid visualization: {grid_save_path.absolute()}")
    print(f"  2. Center volume (standalone): {center_save_path.absolute()}")

    print("\nVisualization Details:")
    print(f"  - Volume shape: {volume.shape}")
    print(f"  - Top row (diffusion): {len(args.timesteps)} timesteps")
    print(f"  - Left column (masking): 6 masking ratios (0-95%)")
    print(f"  - Center volume: 4×4 grid (60% masking + t=0.5 diffusion)")
    print(f"  - SDE type: {args.sde}")
    print(f"  - Transparency: {args.transparency}")
    print(f"  - Image DPI: {args.dpi}")

    print("\n3D dual corruption grid and standalone center volume ready for publication!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
