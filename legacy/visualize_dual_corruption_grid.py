#!/usr/bin/env python3
"""
MDAE Dual Corruption Grid Visualization Script

This script generates a publication-quality grid figure showing MDAE's dual corruption
strategy using real T1-weighted MRI from the m4raw dataset.

The visualization shows:
- Columns: Diffusion timesteps (clean → noisy)
- Rows: Masking ratios (no mask → high mask)
- Highlighted cell: Typical training sample with medium corruption

Usage:
    python visualize_dual_corruption_grid.py

Author: Based on NoiseConditionedMDAETrainer and Corruption2Self implementations
Date: October 2025
"""

import argparse
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Import our utilities
from data_loader_utils import load_m4raw_sample
from plot_utils import setup_cvpr_style, create_dual_corruption_grid


def main():
    parser = argparse.ArgumentParser(
        description="Generate dual corruption grid visualization using m4raw data"
    )
    parser.add_argument(
        '--case-id',
        type=str,
        default=None,
        help='Specific case ID (default: auto-select first available)'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='T1',
        choices=['T1', 'T2', 'FLAIR'],
        help='MRI modality (default: T1)'
    )
    parser.add_argument(
        '--acquisition',
        type=str,
        default=None,
        help='Specific acquisition (e.g., T102). If None, uses first for modality'
    )
    parser.add_argument(
        '--slice-idx',
        type=int,
        default=None,
        help='Which slice to visualize (0-17). If None, uses middle slice'
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
        help='DPI for figures (default: 300)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='both',
        choices=['pdf', 'png', 'both'],
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=16,
        help='Patch size for blocky masking (default: 16)'
    )

    args = parser.parse_args()

    # Setup
    print("=" * 70)
    print("MDAE Dual Corruption Grid Visualization")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    setup_cvpr_style()

    # Load m4raw sample
    print("\n1. Loading m4raw sample...")
    try:
        image_2d, metadata = load_m4raw_sample(
            case_id=args.case_id,
            modality=args.modality,
            acquisition=args.acquisition,
            split='test',
            slice_idx=args.slice_idx,
            cache_dir=str(cache_dir)
        )
        print(f"   ✓ Loaded m4raw sample")
        print(f"   Case ID: {metadata['case_id']}")
        print(f"   Modality: {metadata['modality']}")
        print(f"   Acquisition: {metadata['acquisition']}")
        print(f"   Slice: {metadata['slice_idx']}/{metadata['num_slices']-1}")
        print(f"   Shape: {image_2d.shape}")
    except Exception as e:
        print(f"   ✗ Failed to load m4raw data: {e}")
        print(f"   Please check that the dataset is accessible")
        return 1

    # Generate dual corruption grid
    print("\n2. Generating dual corruption grid...")

    # Define timesteps and masking ratios
    timesteps = np.array([0, 0.005, 0.01, 0.02, 0.5, 1, 2, 3, 5]) / 10
    mask_ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]

    # Highlight cell: row 4 (75% mask), column 4 (t=0.5)
    highlight_cell = (4, 4)

    fig = create_dual_corruption_grid(
        image_2d,
        timesteps=timesteps,
        mask_ratios=mask_ratios,
        patch_size=args.patch_size,
        highlight_cell=highlight_cell,
        save_path=None  # Will save manually with correct format
    )

    # Save in requested format(s)
    base_name = "dual_corruption_grid"
    if args.format in ['pdf', 'both']:
        save_path = output_dir / f"{base_name}.pdf"
        fig.savefig(save_path, dpi=args.dpi, bbox_inches='tight', format='pdf')
        print(f"   ✓ Saved: {save_path}")

    if args.format in ['png', 'both']:
        save_path = output_dir / f"{base_name}.png"
        fig.savefig(save_path, dpi=args.dpi, bbox_inches='tight', format='png')
        print(f"   ✓ Saved: {save_path}")

    plt.close(fig)

    # Summary
    print("\n" + "=" * 70)
    print("✓ Dual corruption grid generated successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"Generated files:")
    for f in output_dir.glob(f"{base_name}.*"):
        print(f"  - {f.name}")

    print("\nVisualization Details:")
    print(f"  - Timesteps: {len(timesteps)} (columns)")
    print(f"  - Masking ratios: {len(mask_ratios)} (rows)")
    print(f"  - Patch size: {args.patch_size}×{args.patch_size}")
    print(f"  - Highlighted cell: Row {highlight_cell[0]}, Column {highlight_cell[1]}")
    print(f"    (Masking: {mask_ratios[highlight_cell[0]]*100:.0f}%, Timestep: {timesteps[highlight_cell[1]]:.2f})")

    print("\nThis figure is ready for CVPR submission!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
