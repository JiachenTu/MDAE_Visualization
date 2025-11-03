#!/usr/bin/env python3
"""
MDAE 3D Dual Corruption Visualization Script using PyVista

This script generates high-quality 3D volume renderings showing the MDAE dual
corruption strategy using PyVista for interactive volume visualization.

Features:
- 3D transparent volume rendering of brain MRI
- 3D visualization of blocky masking patterns
- Side-by-side comparison of corruption stages
- Multi-angle views (axial, sagittal, coronal, isometric)
- Publication-quality static images

Usage:
    # Generate all 3D visualizations
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
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our utilities
from data_loader_utils import load_brats_sample, create_synthetic_brain_volume
from corruption_utils import dual_corruption
from pyvista_utils import (
    setup_pyvista_plotter,
    render_3d_volume,
    render_3d_mask_blocks,
    create_side_by_side_comparison,
    render_multiview,
    create_mask_overlay_3d,
    save_publication_image
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D visualizations of MDAE dual corruption using PyVista"
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
    parser.add_argument(
        '--visualizations',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'volume', 'mask', 'comparison', 'multiview', 'overlay'],
        help='Which visualizations to generate (default: all)'
    )

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print("MDAE 3D Dual Corruption Visualization (PyVista)")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which visualizations to generate
    if 'all' in args.visualizations:
        visualizations = ['volume', 'mask', 'comparison', 'multiview', 'overlay']
    else:
        visualizations = args.visualizations

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

    # Generate visualizations
    print(f"\n3. Generating 3D visualizations...")
    generated_files = []

    # Visualization 1: Clean volume with transparent rendering
    if 'volume' in visualizations:
        print("   → Rendering clean volume (transparent)...")

        plotter = setup_pyvista_plotter(window_size=(1920, 1080), off_screen=True)
        render_3d_volume(
            clean_volume,
            plotter,
            cmap='gray',
            title='Clean MRI Volume',
            camera_position='iso'
        )
        save_path = output_dir / "volume_3d_clean.png"
        save_publication_image(plotter, str(save_path), dpi=args.dpi)
        plotter.close()
        generated_files.append(save_path.name)

    # Visualization 2: 3D blocky mask pattern
    if 'mask' in visualizations:
        print("   → Rendering 3D blocky mask pattern...")
        plotter = setup_pyvista_plotter(window_size=(1920, 1080), off_screen=True)
        render_3d_mask_blocks(
            spatial_mask,
            plotter,
            patch_size=args.patch_size,
            visible_color='lightgreen',
            masked_color='lightcoral',
            opacity=0.7,
            show_visible=True,
            show_masked=True,
            title=f'Blocky Spatial Mask ({args.mask_percentage:.0%} masked)',
            camera_position='iso'
        )
        save_path = output_dir / "volume_3d_mask_pattern.png"
        save_publication_image(plotter, str(save_path), dpi=args.dpi)
        plotter.close()
        generated_files.append(save_path.name)

    # Visualization 3: Side-by-side comparison (6-panel)
    if 'comparison' in visualizations:
        print("   → Creating side-by-side comparison...")
        volumes = [
            clean_volume,
            result['binary_mask_viz'],  # Panel 2: Binary mask (1=masked, 0=visible)
            noisy_volume,
            corrupted_volume,
            clean_volume,  # Panel 5: Clean volume (reconstruction target for masked regions)
            clean_volume   # Panel 6: Clean volume (reconstruction target for visible regions)
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
            result['masked_only_opacity'],  # Masked Regions Only - show MASKED regions opaque
            result['visible_only_opacity']  # Visible Regions Only - show VISIBLE regions opaque
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
        generated_files.append(save_path.name)

    # Visualization 4: Multi-angle views
    if 'multiview' in visualizations:
        print("   → Rendering multi-angle views...")
        plotter = render_multiview(
            clean_volume,
            views=['xy', 'xz', 'yz', 'iso'],
            titles=['Axial', 'Coronal', 'Sagittal', '3D Isometric'],
            window_size=(2400, 2400),
            save_path=None
        )
        save_path = output_dir / "volume_3d_multiview.png"
        save_publication_image(plotter, str(save_path), dpi=args.dpi)
        plotter.close()
        generated_files.append(save_path.name)

    # Visualization 5: Volume with mask overlay
    if 'overlay' in visualizations:
        print("   → Creating volume with mask overlay...")
        plotter = create_mask_overlay_3d(
            clean_volume,
            spatial_mask,
            patch_size=args.patch_size,
            window_size=(1920, 1080),
            save_path=None
        )
        save_path = output_dir / "volume_3d_mask_overlay.png"
        save_publication_image(plotter, str(save_path), dpi=args.dpi)
        plotter.close()
        generated_files.append(save_path.name)

    # Summary
    print("\n" + "=" * 80)
    print("✓ All 3D visualizations generated successfully!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"Generated files ({len(generated_files)}):")
    for f in generated_files:
        print(f"  - {f}")

    print("\nVisualization Details:")
    print(f"  - Volume shape: {volume.shape}")
    print(f"  - Masking ratio: {args.mask_percentage:.0%}")
    print(f"  - Actual masked voxels: {(1 - spatial_mask.float().mean()).item():.1%}")
    print(f"  - Diffusion timestep: {args.timestep:.2f}")
    print(f"  - Patch size: {args.patch_size}³ voxels")
    print(f"  - Number of patches: {(volume.shape[0]//args.patch_size)**3}")
    print(f"  - Image DPI: {args.dpi}")

    print("\nThese 3D visualizations are ready for publication!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
