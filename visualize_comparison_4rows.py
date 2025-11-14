#!/usr/bin/env python3
"""
MDAE 4-Row Comparison Visualization Script

This script generates a 4×6 grid comparison showing different corruption strategies:
- Row 1 (MAE): Masking only (75%), different patterns, no noise
- Row 2 (Diffusion): DDPM noise only at increasing levels, no masking
- Row 3 (DiffMAE): 75% masking + DDPM noise ONLY on masked regions
- Row 4 (MDAE): Progressive dual corruption (masking + noise both increase)

Usage:
    # Generate 4-row comparison with default settings
    python visualize_comparison_4rows.py --dataset brats18 --modality t1ce

    # Use specific case
    python visualize_comparison_4rows.py --case-id Brats18_2013_0_1

    # High-resolution output
    python visualize_comparison_4rows.py --dpi 600

    # Use synthetic data for testing
    python visualize_comparison_4rows.py --use-synthetic

Author: Based on MDAE visualization toolkit
Date: January 2025
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our utilities
from data_loader_utils import load_brats_sample, create_synthetic_brain_volume
from pyvista_utils import create_comparison_4rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate 4-row comparison: MAE vs Diffusion vs DiffMAE vs MDAE"
    )
    parser.add_argument(
        '--sample-number',
        type=int,
        default=5,
        help='Number of corrupted samples per row (default: 5). Total panels per row = 1 (clean) + sample-number'
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
        default='outputs_fig1',
        help='Output directory (default: outputs_fig1)'
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
        '--patch-size',
        type=int,
        default=16,
        help='Patch size for blocky masking (default: 16)'
    )
    parser.add_argument(
        '--mask-percentage',
        type=float,
        default=0.75,
        help='Masking percentage for MAE/DiffMAE rows (default: 0.75)'
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
    parser.add_argument(
        '--corruption-min',
        type=float,
        default=0.2,
        help='Minimum corruption level (default: 0.2)'
    )
    parser.add_argument(
        '--corruption-max',
        type=float,
        default=0.9,
        help='Maximum corruption level (default: 0.9)'
    )

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print("MDAE 4-Row Comparison Visualization")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load volume
    print("\n1. Loading volume...")
    case_name = "synthetic"
    modality_name = "synthetic"

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

            # Store case name and modality for filename
            case_name = metadata['case_id']
            modality_name = metadata['modality']
        except Exception as e:
            print(f"   ✗ Failed to load BraTS data: {e}")
            print(f"   Falling back to synthetic data...")
            args.use_synthetic = True

    if args.use_synthetic:
        volume = create_synthetic_brain_volume(shape=(128, 128, 128), seed=42)
        print(f"   ✓ Created synthetic brain volume")
        print(f"   Shape: {volume.shape}")
        case_name = "synthetic"
        modality_name = "synthetic"

    # Generate 4-row comparison
    print(f"\n2. Generating 4-row comparison visualization...")
    print(f"   Grid: 4 rows × {1 + args.sample_number} columns")
    print(f"   Row 1 (MAE): Masking only ({args.mask_percentage:.0%})")
    print(f"   Row 2 (Diffusion): DDPM noise only ({args.corruption_min}-{args.corruption_max})")
    print(f"   Row 3 (DiffMAE): Masking ({args.mask_percentage:.0%}) + noise on masked ({args.corruption_min}-{args.corruption_max})")
    print(f"   Row 4 (MDAE): Dual corruption ({args.corruption_min}-{args.corruption_max})")
    print(f"   Patch size: {args.patch_size}³")
    print(f"   SDE type: {args.sde}")
    print(f"   Transparency: {args.transparency}")

    # Define save path
    filename = f"MDAE_Comparison_4rows_{case_name}_{modality_name}_{args.sample_number}samples.png"
    comparison_save_path = output_dir / filename

    # Generate corruption levels
    import numpy as np
    corruption_levels = list(np.linspace(args.corruption_min, args.corruption_max, args.sample_number))

    plotter = create_comparison_4rows(
        volume,
        num_corrupted_samples=args.sample_number,
        corruption_levels=corruption_levels,
        mask_percentage=args.mask_percentage,
        patch_size=args.patch_size,
        transparency_level=args.transparency,
        sde=args.sde,
        save_path=str(comparison_save_path)
    )

    # Note: save_publication_image is called inside create_comparison_4rows
    plotter.close()

    # Summary
    print("\n" + "=" * 80)
    print("✓ 4-row comparison visualization generated successfully!")
    print("=" * 80)
    print(f"\nOutput file:")
    print(f"  {comparison_save_path.absolute()}")

    print("\nVisualization Details:")
    print(f"  - Volume shape: {volume.shape}")
    print(f"  - Grid: 4 rows × {1 + args.sample_number} columns")
    print(f"  - Row 1: MAE (masking {args.mask_percentage:.0%}, no noise)")
    print(f"  - Row 2: Diffusion (noise {args.corruption_min}-{args.corruption_max}, no masking)")
    print(f"  - Row 3: DiffMAE (masking {args.mask_percentage:.0%}, noise {args.corruption_min}-{args.corruption_max} on masked)")
    print(f"  - Row 4: MDAE (dual {args.corruption_min}-{args.corruption_max})")
    print(f"  - Patch size: {args.patch_size}³")
    print(f"  - SDE type: {args.sde}")
    print(f"  - Transparency: {args.transparency}")
    print(f"  - Image DPI: {args.dpi}")

    print("\n4-row comparison ready for publication!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
