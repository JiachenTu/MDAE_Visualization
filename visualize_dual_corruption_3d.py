#!/usr/bin/env python3
"""
MDAE 3D Dual Corruption 7-Panel Visualization Script

This script generates a high-quality 7-panel 3D volume rendering showing the
complete MDAE dual corruption pipeline using PyVista, organized in 3 rows.

Layout (3 rows):
Row 0 (2 panels):
  1. Clean Volume - Original MRI volume
  2. Noisy Volume - After diffusion noise corruption

Row 1 (3 panels):
  3. Doubly Corrupted - After both noise and spatial masking
  4. Masked Regions Only - Reconstruction target for masked regions
  5. Visible Regions Only - Visible context for the model

Row 2 (2 panels):
  6. Spatial Mask - Shows masked regions (opaque) and visible regions (transparent)
  7. Visible Mask - Shows visible regions (opaque) and masked regions (transparent)

Usage:
    # Generate 7-panel visualization with default settings
    python visualize_dual_corruption_3d.py

    # Use specific dataset and modality
    python visualize_dual_corruption_3d.py --dataset brats18 --modality t1ce

    # Custom masking and timestep
    python visualize_dual_corruption_3d.py --mask-percentage 0.6 --timestep 0.5

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
import torch
import numpy as np

# Import our utilities
from data_loader_utils import load_brats_sample, create_synthetic_brain_volume
from corruption_utils import dual_corruption
from pyvista_utils import (
    create_side_by_side_comparison,
    save_publication_image,
    setup_pyvista_plotter,
    render_3d_volume
)


def save_individual_panels(
    volumes,
    titles,
    opacity_masks,
    mask_percentage,
    timestep,
    output_dir,
    dpi=300,
    patch_size=16
):
    """
    Save each panel as an individual high-quality figure.

    Args:
        volumes: List of volumes to render
        titles: List of panel titles
        opacity_masks: List of opacity masks (or None)
        mask_percentage: Masking percentage used
        timestep: Diffusion timestep used
        output_dir: Output directory path
        dpi: DPI for output images
        patch_size: Patch size for masking
    """
    from pathlib import Path

    # Create subdirectory for individual panels
    individual_dir = Path(output_dir) / "individual_panels"
    individual_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n4. Saving individual panels to {individual_dir}/")

    # Generate filenames based on panel content and parameters
    visible_pct = int((1 - mask_percentage) * 100)
    mask_pct = int(mask_percentage * 100)
    t_str = f"{timestep:.2f}".replace('.', '')  # e.g., 0.50 -> 050

    filenames = [
        "panel_1_clean_volume.png",
        f"panel_2_noisy_volume_t{timestep:.2f}.png",
        f"panel_3_doubly_corrupted_t{timestep:.2f}_m{mask_pct}pct.png",
        f"panel_4_masked_regions_only_m{mask_pct}pct.png",
        f"panel_5_visible_regions_only_m{visible_pct}pct.png",
        f"panel_6_spatial_mask_m{mask_pct}pct.png",
        f"panel_7_visible_mask_m{visible_pct}pct.png"
    ]

    # Render and save each panel
    for idx, (volume, title, filename) in enumerate(zip(volumes, titles, filenames)):
        # Create single-panel plotter with larger window for high quality
        plotter = setup_pyvista_plotter(
            window_size=(3600, 3600),  # High-quality square window
            off_screen=True,
            shape=(1, 1)
        )

        # Get opacity mask for this panel
        opacity_mask = opacity_masks[idx] if idx < len(opacity_masks) else None

        # Render with same settings as combined view, but NO TITLE
        if 'spatial mask' in title.lower() or 'visible mask' in title.lower():
            # Mask panels use special rendering
            render_3d_volume(
                volume,
                plotter,
                transparency_level='medium',
                cmap='Blues',
                title='',  # No title for individual panels
                camera_position='iso',
                clim=(0, 1),
                opacity_mask=opacity_mask,
                use_uniform_opacity=True
            )
        else:
            # Other panels use standard rendering
            if 'visible regions only' in title.lower() or 'masked regions only' in title.lower():
                trans_level = 'medium'
            else:
                trans_level = 'medium'

            render_3d_volume(
                volume,
                plotter,
                transparency_level=trans_level,
                cmap='gray',
                title='',  # No title for individual panels
                camera_position='iso',
                opacity_mask=opacity_mask
            )

        # Save panel with high quality
        save_path = individual_dir / filename
        save_publication_image(plotter, str(save_path), dpi=dpi)
        plotter.close()

        print(f"   ✓ Saved: {filename}")

    print(f"   Total: {len(volumes)} individual panels saved")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 7-panel 3D visualization of MDAE dual corruption pipeline (3-row layout)"
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
        '--save-individual',
        action='store_true',
        help='Save each panel as individual high-quality figure in subdirectory'
    )

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print("MDAE 3D Dual Corruption 7-Panel Visualization (3-Row Layout)")
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

    # Generate 7-panel visualization with 3-row layout
    print(f"\n3. Generating 7-panel 3D visualization (3 rows)...")
    print("   → Row 0: Clean Volume, Noisy Volume")
    print("   → Row 1: Doubly Corrupted, Masked Regions Only, Visible Regions Only")
    print("   → Row 2: Spatial Mask, Visible Mask")

    # Create inverted mask for Spatial Mask panel
    # binary_mask_viz: 1=visible, 0=masked
    # inverted_mask_viz: 1=masked, 0=visible (so masked regions show as bright blue)
    if isinstance(result['binary_mask_viz'], torch.Tensor):
        inverted_mask_viz = 1.0 - result['binary_mask_viz']
    else:
        inverted_mask_viz = 1.0 - result['binary_mask_viz']

    volumes = [
        # Row 0
        clean_volume,                      # [0,0]
        noisy_volume,                      # [0,1]
        # Row 1
        corrupted_volume,                  # [1,0]
        result['masked_regions_viz'],      # [1,1]
        result['visible_regions_viz'],     # [1,2]
        # Row 2
        inverted_mask_viz,                 # [2,0] Spatial Mask (1=masked, shows bright blue for masked)
        result['binary_mask_viz']          # [2,1] Visible Mask (1=visible, shows bright blue for visible)
    ]
    titles = [
        # Row 0
        'Clean Volume',
        f'Noisy Volume (t={args.timestep:.2f})',
        # Row 1
        'Doubly Corrupted',
        'Masked Regions Only',
        'Visible Regions Only',
        # Row 2
        f'Spatial Mask ({args.mask_percentage:.0%} masked)',
        f'Visible Mask ({(1-args.mask_percentage):.0%} visible)'
    ]
    # Create opacity masks for each panel
    opacity_masks = [
        # Row 0
        None,                              # Clean Volume - default rendering
        None,                              # Noisy Volume - default rendering
        # Row 1
        None,                              # Doubly Corrupted - default rendering
        result['masked_only_opacity'],     # Masked Regions Only - show MASKED regions only
        result['visible_only_opacity'],    # Visible Regions Only - show VISIBLE regions only
        # Row 2
        None,                              # Spatial Mask - show full mask pattern (no opacity masking)
        None                               # Visible Mask - show full mask pattern (no opacity masking)
    ]
    # Define custom panel positions in 3×3 grid
    panel_positions = [
        (0, 0),  # Clean Volume
        (0, 1),  # Noisy Volume
        (1, 0),  # Doubly Corrupted
        (1, 1),  # Masked Regions Only
        (1, 2),  # Visible Regions Only
        (2, 0),  # Spatial Mask
        (2, 1)   # Visible Mask
    ]

    plotter = create_side_by_side_comparison(
        volumes,
        titles,
        patch_size=args.patch_size,
        window_size=(3600, 3600),  # Square for 3×3 grid
        save_path=None,
        opacity_masks=opacity_masks,
        grid_shape=(3, 3),           # 3 rows × 3 columns
        panel_positions=panel_positions
    )

    save_path = output_dir / "volume_3d_comparison_7panel.png"
    save_publication_image(plotter, str(save_path), dpi=args.dpi)
    plotter.close()

    # Save individual panels if requested
    if args.save_individual:
        save_individual_panels(
            volumes=volumes,
            titles=titles,
            opacity_masks=opacity_masks,
            mask_percentage=args.mask_percentage,
            timestep=args.timestep,
            output_dir=output_dir,
            dpi=args.dpi,
            patch_size=args.patch_size
        )

    # Summary
    print("\n" + "=" * 80)
    print("✓ 7-panel visualization generated successfully!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  1. Combined visualization: {save_path.absolute()}")
    if args.save_individual:
        individual_dir = output_dir / "individual_panels"
        print(f"  2. Individual panels (7): {individual_dir.absolute()}/")

    print("\nVisualization Layout:")
    print(f"  Row 0 (2 cols): Clean Volume | Noisy Volume")
    print(f"  Row 1 (3 cols): Doubly Corrupted | Masked Regions Only | Visible Regions Only")
    print(f"  Row 2 (2 cols): Spatial Mask (masked regions) | Visible Mask (visible regions)")

    print("\nVisualization Details:")
    print(f"  - Volume shape: {volume.shape}")
    print(f"  - Masking ratio: {args.mask_percentage:.0%}")
    print(f"  - Actual masked voxels: {(1 - spatial_mask.float().mean()).item():.1%}")
    print(f"  - Diffusion timestep: {args.timestep:.2f}")
    print(f"  - Patch size: {args.patch_size}³ voxels")
    print(f"  - Number of patches: {(volume.shape[0]//args.patch_size)**3}")
    print(f"  - Grid layout: 3×3 (7 panels used, 2 empty)")
    print(f"  - Image DPI: {args.dpi}")

    if args.save_individual:
        print("\n7-panel visualization + individual panels ready for publication!")
    else:
        print("\n7-panel visualization with mask comparison ready for publication!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
