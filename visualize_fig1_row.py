#!/usr/bin/env python3
"""
MDAE Figure 1 Visualization Script

This script generates a horizontal row of 3D volume renderings showing the progression
from clean brain MRI to increasingly corrupted MDAE samples.

The visualization shows:
- Leftmost panel: Clean volume (no corruption)
- Remaining panels: MDAE dual-corrupted volumes with increasing masking ratio AND noise level

Both corruption parameters (masking ratio and diffusion timestep) progress together
linearly from 0.0 to 0.95, demonstrating the full range of MDAE's dual corruption strategy.

Usage:
    # Generate row with 5 corrupted samples (6 panels total)
    python visualize_fig1_row.py

    # Generate row with 3 corrupted samples (4 panels total)
    python visualize_fig1_row.py --sample-number 3

    # Use specific dataset and modality
    python visualize_fig1_row.py --dataset brats18 --modality t1ce

    # Use specific case
    python visualize_fig1_row.py --case-id Brats18_2013_0_1

    # High-resolution output
    python visualize_fig1_row.py --dpi 600

    # Use synthetic data for testing
    python visualize_fig1_row.py --use-synthetic

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
from pyvista_utils import create_fig1_row


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 1: horizontal row showing clean to corrupted progression"
    )
    parser.add_argument(
        '--sample-number',
        type=int,
        default=5,
        help='Number of corrupted samples (default: 5). Total panels = 1 (clean) + sample-number'
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
        default=0.0,
        help='Minimum corruption level (default: 0.0)'
    )
    parser.add_argument(
        '--corruption-max',
        type=float,
        default=0.95,
        help='Maximum corruption level (default: 0.95)'
    )

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print("MDAE Figure 1 Visualization")
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

    # Generate Figure 1 row visualization
    print(f"\n2. Generating Figure 1 row visualization...")
    total_panels = 1 + args.sample_number
    print(f"   Total panels: {total_panels} (1 clean + {args.sample_number} corrupted)")
    print(f"   Corruption progression: {args.corruption_min} → {args.corruption_max} (both masking and noise)")
    print(f"   Patch size: {args.patch_size}³")
    print(f"   SDE type: {args.sde}")
    print(f"   Transparency: {args.transparency}")

    # Define save path with case name and metadata
    # Format: MDAE_[case_name]_[modality]_[N]samples_[min]-[max].png
    filename = f"MDAE_{case_name}_{modality_name}_{args.sample_number}samples_{args.corruption_min}-{args.corruption_max}.png"
    fig1_save_path = output_dir / filename

    plotter = create_fig1_row(
        volume,
        num_corrupted_samples=args.sample_number,
        patch_size=args.patch_size,
        transparency_level=args.transparency,
        sde=args.sde,
        corruption_min=args.corruption_min,
        corruption_max=args.corruption_max,
        save_path=str(fig1_save_path)
    )

    # Note: save_publication_image is called inside create_fig1_row
    plotter.close()

    # Summary
    print("\n" + "=" * 80)
    print("✓ Figure 1 visualization generated successfully!")
    print("=" * 80)
    print(f"\nOutput file:")
    print(f"  {fig1_save_path.absolute()}")

    print("\nVisualization Details:")
    print(f"  - Volume shape: {volume.shape}")
    print(f"  - Total panels: {total_panels}")
    print(f"  - Clean volume: Panel 0")
    print(f"  - Corrupted samples: Panels 1-{args.sample_number}")
    print(f"  - Corruption range: {args.corruption_min} to {args.corruption_max}")
    print(f"  - Patch size: {args.patch_size}³")
    print(f"  - SDE type: {args.sde}")
    print(f"  - Transparency: {args.transparency}")
    print(f"  - Image DPI: {args.dpi}")

    print("\nFigure 1 visualization ready for publication!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
