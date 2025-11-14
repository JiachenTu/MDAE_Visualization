#!/usr/bin/env python3
"""
Simple Diffusion Corruption Visualization

Creates a single row showing the diffusion corruption progression from clean to noisy.
Minimal design with only essential labels.

Usage:
    python visualize_diffusion_row.py [--case-id 2022101102]
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data_loader_utils import load_m4raw_sample
from corruption_utils import corrupt_ddpm
from plot_utils import setup_cvpr_style


def main():
    parser = argparse.ArgumentParser(description="Simple diffusion corruption visualization")
    parser.add_argument('--case-id', type=str, default='2022101101',
                       help='M4Raw case ID (default: 2022101101)')
    parser.add_argument('--modality', type=str, default='T1', choices=['T1', 'T2', 'FLAIR'],
                       help='MRI modality (default: T1)')
    parser.add_argument('--acquisition', type=str, default='T102',
                       help='MRI acquisition (default: T102 for better quality)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for output (default: 300)')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use default matplotlib style for better contrast (like the notebook)
    # setup_cvpr_style()  # Commented out to match notebook's appearance

    # Load sample (using T102 acquisition for better image quality, matching notebook)
    print(f"Loading M4Raw sample {args.case_id}...")
    image_2d, metadata = load_m4raw_sample(
        case_id=args.case_id,
        modality=args.modality,
        acquisition=args.acquisition,
        cache_dir='sample_data'
    )
    print(f"Loaded: {metadata['case_id']} - {metadata['modality']} - Slice {metadata['slice_idx']}")

    # Timesteps (matching notebook)
    timesteps = np.array([0, 0.005, 0.01, 0.02, 0.5, 1, 3, 5, 10]) / 10

    # Generate corrupted images
    print(f"Generating {len(timesteps)} corrupted versions...")
    noisy_images = []
    for t in timesteps:
        noisy = corrupt_ddpm(image_2d, t)
        noisy_images.append(noisy.cpu().numpy())

    # Create simple single-row plot
    fig, axes = plt.subplots(1, len(noisy_images), figsize=(18, 2.5))

    for i, ax in enumerate(axes):
        ax.imshow(noisy_images[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    # Add minimal labels on key timesteps only
    axes[0].set_title(r'$(\mathbf{X}_0, 0)$', fontsize=12, pad=10)
    axes[2].set_title(r'$(\mathbf{X}_{t_{\mathrm{data}}}, t_{\mathrm{data}})$', fontsize=12, pad=10)
    axes[6].set_title(r'$(\mathbf{X}_{\tau}, \tau)$', fontsize=12, pad=10)
    axes[8].set_title(r'$(\mathbf{X}_T, T)$', fontsize=12, pad=10)

    plt.tight_layout()

    # Save
    pdf_path = output_dir / 'diffusion_corruption_row.pdf'
    png_path = output_dir / 'diffusion_corruption_row.png'

    fig.savefig(pdf_path, dpi=args.dpi, bbox_inches='tight', format='pdf')
    fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight', format='png')

    print(f"\n✓ Saved:")
    print(f"  - {pdf_path}")
    print(f"  - {png_path}")

    plt.close()


if __name__ == "__main__":
    main()
