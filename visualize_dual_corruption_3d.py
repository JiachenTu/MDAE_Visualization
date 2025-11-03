#!/usr/bin/env python3
"""
MDAE 3D Dual Corruption 6-Panel Visualization Script

This script generates a high-quality 6-panel 3D volume rendering showing the
complete MDAE dual corruption pipeline using PyVista.

The 6 panels show:
1. Clean Volume - Original MRI volume
2. Spatial Mask - Blocky masking pattern (75% masked)
3. Noisy Volume - After diffusion noise corruption
4. Doubly Corrupted - After both noise and spatial masking
5. Masked Regions Only - Reconstruction target for masked regions
6. Visible Regions Only - Visible context for the model

Usage:
    # Generate 6-panel visualization with default settings
    python visualize_dual_corruption_3d.py

    # Use specific dataset and modality
    python visualize_dual_corruption_3d.py --dataset brats18 --modality t1ce

    # High-resolution output
    python visualize_dual_corruption_3d.py --dpi 600

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
from corruption_utils import dual_corruption
from pyvista_utils import (
    create_side_by_side_comparison,
    save_publication_image
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 6-panel 3D visualization of MDAE dual corruption pipeline"
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
        '--mask-percentage',
        type=float,
        default=0.75,
        help='Masking percentage (default: 0.75)'
    )
    parser.add_argument(
        '--timestep',
        type=float,
        default=0.5,
        help='Diffusion timestep (default: 0.5)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=16,
        help='Patch size for blocky masking (default: 16)'
    )

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print("MDAE 3D Dual Corruption 6-Panel Visualization")
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

    # Apply dual corruption
    print(f"\n2. Applying dual corruption...")
    print(f"   Mask percentage: {args.mask_percentage:.0%}")
    print(f"   Diffusion timestep: {args.timestep:.2f}")
    print(f"   Patch size: {args.patch_size}³")

    result = dual_corruption(
        volume,
        mask_percentage=args.mask_percentage,
        timestep=args.timestep,
        patch_size=args.patch_size
    )

    clean_volume = result['clean_volume']
    spatial_mask = result['spatial_mask']
    noisy_volume = result['noisy_volume']
    corrupted_volume = result['doubly_corrupted']

    print(f"   ✓ Dual corruption applied")
    print(f"   Alpha(t): {result['alpha_t']:.4f}, Sigma(t): {result['sigma_t']:.4f}")

    # Generate 6-panel visualization
    print(f"\n3. Generating 6-panel 3D visualization...")
    print("   → Creating 6-panel comparison...")

    volumes = [
        clean_volume,
        result['binary_mask_viz'],  # Panel 2: Binary mask (1=masked, 0=visible)
        noisy_volume,
        corrupted_volume,
        result['masked_regions_viz'],  # Panel 5: Clean with dimmed visible regions (gradient fade)
        result['visible_regions_viz']  # Panel 6: Clean with dimmed masked regions (gradient fade)
    ]
    titles = [
        'Clean Volume',
        f'Spatial Mask ({args.mask_percentage:.0%})',
        f'Noisy Volume (t={args.timestep:.2f})',
        'Doubly Corrupted',
        'Masked Regions Only',
        'Visible Regions Only'
    ]
    # Create opacity masks for panels 2, 5, and 6
    opacity_masks = [
        None,  # Clean Volume - default rendering
        result['masked_only_opacity'],  # Spatial Mask - show MASKED regions opaque, visible transparent
        None,  # Noisy Volume - default rendering
        None,  # Doubly Corrupted - default rendering
        result['masked_only_opacity'],  # Masked Regions Only - show MASKED regions, transparent visible regions
        result['visible_only_opacity']  # Visible Regions Only - show VISIBLE regions, transparent masked regions
    ]

    plotter = create_side_by_side_comparison(
        volumes,
        titles,
        patch_size=args.patch_size,
        window_size=(3600, 2400),  # Wider for 3 columns
        save_path=None,
        opacity_masks=opacity_masks
    )

    save_path = output_dir / "volume_3d_comparison_6panel.png"
    save_publication_image(plotter, str(save_path), dpi=args.dpi)
    plotter.close()

    # Summary
    print("\n" + "=" * 80)
    print("✓ 6-panel visualization generated successfully!")
    print("=" * 80)
    print(f"\nOutput file: {save_path.absolute()}")

    print("\nVisualization Details:")
    print(f"  - Volume shape: {volume.shape}")
    print(f"  - Masking ratio: {args.mask_percentage:.0%}")
    print(f"  - Actual masked voxels: {(1 - spatial_mask.float().mean()).item():.1%}")
    print(f"  - Diffusion timestep: {args.timestep:.2f}")
    print(f"  - Patch size: {args.patch_size}³ voxels")
    print(f"  - Number of patches: {(volume.shape[0]//args.patch_size)**3}")
    print(f"  - Image DPI: {args.dpi}")

    print("\n6-panel visualization is ready for publication!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
